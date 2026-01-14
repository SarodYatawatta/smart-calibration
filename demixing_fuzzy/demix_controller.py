import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as fcontrol
import json

class DemixController:
    """
    Fuzzy controller used in demixing
    """
    def __init__(self,n_action):
        """
        n_action: number of actions (input to the controller) = 32
        """
        self.n_action=n_action
        self.config,self.n_var=self.create_defaults()
        assert(self.n_action==self.n_var)
        self.fuzzy_ctrl=None

    def create_defaults(self):
        """
        return (dict with default parameters, number of variables)
        """
        n_var=0
        azimuth=dict() # azimuth (deg)
        azimuth['range']=[-180, 180, 1] # fixed
        azimuth['low']=[-180, -180, -65, -55] # var 2,3
        azimuth['medium']=[-65, -55, 55, 65] # var 2,3
        azimuth['high']=[55, 65, 180, 180] # fixed
        n_var +=4
        azimuth_target=dict() # elevation (deg)
        azimuth_target['range']=[-180, 180, 1] # fixed
        azimuth_target['low']=[-180, -180, -65, -55] # var 2,3
        azimuth_target['medium']=[-65, -55, 55, 65] # var 2,3
        azimuth_target['high']=[55, 65, 180, 180] # fixed
        n_var +=4
 
        elevation=dict() # elevation (deg)
        elevation['range']=[-90, 90, 1] # fixed
        elevation['low']=[-90, -90, -5, 5] # var 2,3
        elevation['medium']=[-5, 5, 50, 60] # var 2,3
        elevation['high']=[50, 60, 90, 90] # fixed
        n_var +=4
        elevation_target=dict() # elevation (deg)
        elevation_target['range']=[-90, 90, 1] # fixed
        elevation_target['low']=[-90, -90, -5, 5] # var 2,3
        elevation_target['medium']=[-5, 5, 50, 60] # var 2,3
        elevation_target['high']=[50, 60, 90, 90] # fixed
        n_var +=4

        separation=dict() # separation (deg)
        separation['range']=[0, 180, 1] # fixed
        separation['low']=[0, 0, 10, 15] # var 2,3
        separation['medium']=[10, 15, 45, 50] # var 2,3
        separation['high']=[45, 50, 180, 180] # fixed
        n_var +=4
        logI=dict() # log flux density 
        logI['range']=[0, 100, 1] # fixed
        logI['low']=[0, 0, 1.0, 2.0] # var 2,3
        logI['medium']=[1.0, 2.0, 5.0, 10] # var 2,3
        logI['high']=[5.0, 10, 100, 100] # fixed
        n_var +=4
        ratI=dict() # flux ratio source/target
        ratI['range']=[0, 100, 1] # fixed
        ratI['low']=[0, 0, 0.5, 1.0] # var 2,3
        ratI['medium']=[0.5, 1.0, 50, 55] # var 2,3
        ratI['high']=[50, 55, 100, 100] # fixed
        n_var +=4

        priority=dict() # priority 0..100
        priority['range']=[0,100,1] # fixed
        priority['low']=[0, 0, 40, 50] # var 2,3
        priority['medium']=[40, 50, 70, 75] # var 2,3
        priority['high']=[70, 75, 100, 100] # fixed
        n_var +=4

        input_limits=dict()
        input_limits['_azimuth']=azimuth
        input_limits['_azimuth_target']=azimuth_target
        input_limits['_elevation']=elevation
        input_limits['_elevation_target']=elevation_target
        input_limits['_separation']=separation
        input_limits['_log_intensity']=logI
        input_limits['_intensity_ratio']=ratI

        output_limits=dict()
        output_limits['_priority']=priority

        membership_limits=dict()
        membership_limits['inputs']=input_limits
        membership_limits['outputs']=output_limits
        membership_limits['_comment']="This file defines the input/output limits, ranges etc. used by tune_demixing_parameters.py. This is automatically generated."

        return membership_limits,n_var

    def update_set_(self,fuzzy_set,action):
        """
        helper function to update any given membership function
        fuzzy_set: low, med, high : [v0, v1, v2, v3] 
        updated to [v0, v1, v2 <= v1+val1*(upper_lim-v1), v3 <= v2+val2*(upper_lim-v2)]
        val1,val2,.. taken from action
        making sure that updated values do not exceed the limits
        """
        upper_lim=fuzzy_set['range'][1]
        fuzzy_set['low'][2]=fuzzy_set['low'][1]+action[0]*(upper_lim-fuzzy_set['low'][1])
        fuzzy_set['low'][3]=fuzzy_set['low'][2]+action[1]*(upper_lim-fuzzy_set['low'][2])
        fuzzy_set['medium'][0]=fuzzy_set['low'][2]
        fuzzy_set['medium'][1]=fuzzy_set['low'][3]
        fuzzy_set['medium'][2]=fuzzy_set['medium'][1]+action[2]*(upper_lim-fuzzy_set['medium'][1])
        fuzzy_set['medium'][3]=fuzzy_set['medium'][2]+action[3]*(upper_lim-fuzzy_set['medium'][2])
        fuzzy_set['high'][0]=fuzzy_set['medium'][2]
        fuzzy_set['high'][1]=fuzzy_set['medium'][3]

    def update_action_(self,fuzzy_set,action):
        """
        helper function to update action based on membership limits
        inverse of update_set_()
        """
        upper_lim=fuzzy_set['range'][1]
        action[0]=(fuzzy_set['low'][2]-fuzzy_set['low'][1])/(upper_lim-fuzzy_set['low'][1])
        action[1]=(fuzzy_set['low'][3]-fuzzy_set['low'][2])/(upper_lim-fuzzy_set['low'][2])
        action[2]=(fuzzy_set['medium'][2]-fuzzy_set['medium'][1])/(upper_lim-fuzzy_set['medium'][1])
        action[3]=(fuzzy_set['medium'][3]-fuzzy_set['medium'][2])/(upper_lim-fuzzy_set['medium'][2])

    def update_limits(self,action):
        """
        update model
        if any action makes the model parameters exceed range, add penalty

        action: n_var x 1 parameter vector in [0,1]
        last 8 are for the target (reused)
        """
        assert(action.size==self.n_var)
        inputs=self.config['inputs']
        outputs=self.config['outputs']

        self.update_set_(inputs['_azimuth'],action[0:4])
        self.update_set_(inputs['_elevation'],action[4:8])
        self.update_set_(inputs['_separation'],action[8:12])
        self.update_set_(inputs['_log_intensity'],action[12:16])
        self.update_set_(inputs['_intensity_ratio'],action[16:20])
        self.update_set_(outputs['_priority'],action[20:24])
        # last 8 actions are for the target
        self.update_set_(inputs['_azimuth_target'],action[24:28])
        self.update_set_(inputs['_elevation_target'],action[28:32])

    def update_action(self):
        """
        update action according to current membership limits
        return action n_var x 1 in [0,1]
        """
        action=np.zeros(self.n_var)
        inputs=self.config['inputs']
        outputs=self.config['outputs']

        self.update_action_(inputs['_azimuth'],action[0:4])
        self.update_action_(inputs['_elevation'],action[4:8])
        self.update_action_(inputs['_separation'],action[8:12])
        self.update_action_(inputs['_log_intensity'],action[12:16])
        self.update_action_(inputs['_intensity_ratio'],action[16:20])
        self.update_action_(outputs['_priority'],action[20:24])
        self.update_action_(inputs['_azimuth_target'],action[24:28])
        self.update_action_(inputs['_elevation_target'],action[28:32])

        return action

    def create_controller(self):
        """
        using the membership limits, create fuzzy controller
        """
        # Define the antecedents using ranges
        azimuth = fcontrol.Antecedent(np.arange(*self.config['inputs']['_azimuth']['range']), 'azimuth')
        azimuth_target = fcontrol.Antecedent(np.arange(*self.config['inputs']['_azimuth_target']['range']), 'azimuth_target')
        elevation = fcontrol.Antecedent(np.arange(*self.config['inputs']['_elevation']['range']), 'elevation')
        elevation_target = fcontrol.Antecedent(np.arange(*self.config['inputs']['_elevation_target']['range']), 'elevation_target')
        separation= fcontrol.Antecedent(np.arange(*self.config['inputs']['_separation']['range']), 'separation')
        log_intensity = fcontrol.Antecedent(np.arange(*self.config['inputs']['_log_intensity']['range']), 'log_intensity')
        intensity_ratio= fcontrol.Antecedent(np.arange(*self.config['inputs']['_intensity_ratio']['range']), 'intensity_ratio')

        # Define the consequents using ranges
        priority = fcontrol.Consequent(np.arange(*self.config['outputs']['_priority']['range']), 'priority')
  
        # Define membership functions for inputs using config
        for var_name, var in zip(['_azimuth', '_azimuth_target', '_elevation', '_elevation_target', '_separation', '_log_intensity', '_intensity_ratio'], [azimuth, azimuth_target, elevation, elevation_target, separation, log_intensity, intensity_ratio]):
            for term, limits in self.config['inputs'][var_name].items():
               if term != 'range':
                  var[term] = fuzz.trapmf(var.universe, limits)

        # Define membership functions for outputs using config
        for var_name, var in zip(['_priority'], [priority]):
            for term, limits in self.config['outputs'][var_name].items():
               if term != 'range':
                  var[term] = fuzz.trapmf(var.universe, limits)

        # Define fuzzy rules - see the documentation for further explanation
        rule0 = fcontrol.Rule(azimuth['low'] & azimuth_target['low'], priority['medium'])
        rule1 = fcontrol.Rule(azimuth['medium'] & azimuth_target['medium'], priority['medium'])
        rule2 = fcontrol.Rule(azimuth['high'] & azimuth_target['high'], priority['medium'])
        rule3 = fcontrol.Rule(separation['low'], priority['high'])
        rule4 = fcontrol.Rule(elevation['low'], priority['low'])
        rule5 = fcontrol.Rule(elevation['low'] & separation['high'] & log_intensity['low'] & intensity_ratio['low'], priority['low'])
        rule6 = fcontrol.Rule(elevation['medium'] &  separation['medium'] & intensity_ratio['high'], priority['medium'])
        rule7 = fcontrol.Rule(elevation['high'] & separation['medium'] & intensity_ratio['high'], priority['high'])
        rule8 = fcontrol.Rule(elevation['high'] & log_intensity['high'] & intensity_ratio['high'], priority['high'])
        rule9 = fcontrol.Rule(elevation['medium'] | separation['medium'] | log_intensity['medium'] | intensity_ratio['medium'], priority['medium'])
        rule10 = fcontrol.Rule(elevation_target['low'] & elevation['high'], priority['high'])
        rule11 = fcontrol.Rule(elevation_target['high'] & elevation['low'], priority['low'])

        system=fcontrol.ControlSystem([rule0,rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11])
        # Flush cache after ~ n_directions evaluations
        self.fuzzy_ctrl=fcontrol.ControlSystemSimulation(system,flush_after_run=7)

    def evaluate(self,azimuth,azimuth_target,elevation,elevation_target,separation,log_intensity,intensity_ratio):
        """
        evaluate the controller to create fuzzy outputs
        azimuth(_target),elevation(_target),separation,log_intensity,intensity_ratio are 
        per each direction (not vectors)
        """
        self.fuzzy_ctrl.input['azimuth']=azimuth
        self.fuzzy_ctrl.input['azimuth_target']=azimuth_target
        self.fuzzy_ctrl.input['elevation']=elevation
        self.fuzzy_ctrl.input['elevation_target']=elevation_target
        self.fuzzy_ctrl.input['separation']=separation
        self.fuzzy_ctrl.input['log_intensity']=log_intensity
        self.fuzzy_ctrl.input['intensity_ratio']=intensity_ratio
        self.fuzzy_ctrl.compute()
        # try to handle delays in compute() and output is missing
        try:
              priority=self.fuzzy_ctrl.output['priority']
        except KeyError:
              print('Warning: compute() fail, using fallback')
              priority=50

        return priority

    def get_high_priority(self):
        """
        return 'high' set priority cutoff, using first value
        """
        return self.config['outputs']['_priority']['high'][0]

    def print_config(self,filename=None):
        """
        export config to JSON, or print to stdout
        """
        if filename:
           with open(filename,'w+') as json_file:
              json.dump(self.config,json_file)
        else:
           print(self.config)


#fuz=DemixController(n_action=32) # actions used per-direction
#K=6
## total actions required (24 per each outlier dir, 8 for target)
#n_act=24*(K-1)+8
#action=np.random.rand(n_act)
#azimuth=(np.random.rand(K)-0.5)*360
#elevation=np.random.rand(K)*90
#separation=np.random.rand(K-1)*90
#log_intensity=np.random.rand(K-1)*3
#intensity_ratio=np.random.rand(K-1)*2
#for ci in range(K-1):
#  # assemble action
#  a=np.zeros(32)
#  a[:24]=action[ci*24:(ci+1)*24]
#  a[-8:]=action[-8:]
#  fuz.update_limits(a)
#  fuz.create_controller()
#  priority=fuz.evaluate(azimuth[ci],azimuth[-1],elevation[ci],elevation[-1],separation[ci],log_intensity[ci],intensity_ratio[ci])
#  print(priority)
#  b=fuz.update_action()
#fuz.print_config(filename='pp')
