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
        n_action:16 
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
        elevation=dict() # elevation (deg)
        elevation['range']=[-90, 90, 1] # fixed
        elevation['low']=[-90, -90, -5, 5] # var 2,3
        elevation['medium']=[-5, 5, 50, 60] # var 2,3
        elevation['high']=[50, 60, 90, 90] # fixed
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
        input_limits['_elevation']=elevation
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
        """
        assert(action.size==self.n_var)
        inputs=self.config['inputs']
        outputs=self.config['outputs']

        self.update_set_(inputs['_elevation'],action[0:4])
        self.update_set_(inputs['_separation'],action[4:8])
        self.update_set_(inputs['_log_intensity'],action[8:12])
        self.update_set_(inputs['_intensity_ratio'],action[12:16])
        self.update_set_(outputs['_priority'],action[16:20])

    def update_action(self):
        """
        update action according to current membership limits
        return action n_var x 1 in [0,1]
        """
        action=np.zeros(self.n_var)
        inputs=self.config['inputs']
        outputs=self.config['outputs']

        self.update_action_(inputs['_elevation'],action[0:4])
        self.update_action_(inputs['_separation'],action[4:8])
        self.update_action_(inputs['_log_intensity'],action[8:12])
        self.update_action_(inputs['_intensity_ratio'],action[12:16])
        self.update_action_(outputs['_priority'],action[16:20])

        return action

    def create_controller(self):
        """
        using the membership limits, create fuzzy controller
        """
        # Define the antecedents using ranges
        elevation = fcontrol.Antecedent(np.arange(*self.config['inputs']['_elevation']['range']), 'elevation')
        separation= fcontrol.Antecedent(np.arange(*self.config['inputs']['_separation']['range']), 'separation')
        log_intensity = fcontrol.Antecedent(np.arange(*self.config['inputs']['_log_intensity']['range']), 'log_intensity')
        intensity_ratio= fcontrol.Antecedent(np.arange(*self.config['inputs']['_intensity_ratio']['range']), 'intensity_ratio')

        # Define the consequents using ranges
        priority = fcontrol.Consequent(np.arange(*self.config['outputs']['_priority']['range']), 'priority')
  
        # Define membership functions for inputs using config
        for var_name, var in zip(['_elevation', '_separation', '_log_intensity', '_intensity_ratio'], [elevation, separation, log_intensity, intensity_ratio]):
            for term, limits in self.config['inputs'][var_name].items():
               if term != 'range':
                  var[term] = fuzz.trapmf(var.universe, limits)

        # Define membership functions for outputs using config
        for var_name, var in zip(['_priority'], [priority]):
            for term, limits in self.config['outputs'][var_name].items():
               if term != 'range':
                  var[term] = fuzz.trapmf(var.universe, limits)

        # Define fuzzy rules - see the documentation for further explanation
        rule1 = fcontrol.Rule(separation['low'], priority['high'])
        rule7 = fcontrol.Rule(elevation['low'], priority['low'])
        rule4 = fcontrol.Rule(elevation['low'] & separation['high'] & log_intensity['low'] & intensity_ratio['low'], priority['low'])
        rule6 = fcontrol.Rule(elevation['medium'] &  separation['medium'] & intensity_ratio['high'], priority['medium'])
        rule3 = fcontrol.Rule(elevation['high'] & separation['medium'] & intensity_ratio['high'], priority['high'])
        rule2 = fcontrol.Rule(elevation['high'] & log_intensity['high'] & intensity_ratio['high'], priority['high'])
        rule5 = fcontrol.Rule(elevation['medium'] | separation['medium'] | log_intensity['medium'] | intensity_ratio['medium'], priority['medium'])

        system=fcontrol.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7])
        # Flush cache after ~ n_directions evaluations
        self.fuzzy_ctrl=fcontrol.ControlSystemSimulation(system,flush_after_run=7)

    def evaluate(self,elevation,separation,log_intensity,intensity_ratio):
        """
        evaluate the controller to create fuzzy outputs
        """
        # total size includes target, get outlier directions
        n_dir=elevation.size-1
        priority=np.zeros(n_dir,dtype=np.float32)
        for ci in range(n_dir):
          self.fuzzy_ctrl.input['elevation']=elevation[ci]
          self.fuzzy_ctrl.input['separation']=separation[ci]
          self.fuzzy_ctrl.input['log_intensity']=log_intensity[ci]
          self.fuzzy_ctrl.input['intensity_ratio']=intensity_ratio[ci]
          self.fuzzy_ctrl.compute()
          # try to handle delays in compute() and output is missing
          try:
              priority[ci]=self.fuzzy_ctrl.output['priority']
          except KeyError:
              print('Warning: compute() fail, using fallback')
              priority[ci]=50

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


#fuz=DemixController(n_action=20)
#fuz.create_controller()
#elevation=np.random.rand(7)*90
#separation=np.random.rand(7)*90
#log_intensity=np.random.rand(7)*3
#intensity_ratio=np.random.rand(7)*2
#priority=fuz.evaluate(elevation,separation,log_intensity,intensity_ratio)
#print(priority)
#a=np.random.rand(20)
#fuz.update_limits(a)
#fuz.create_controller()
#priority=fuz.evaluate(elevation,separation,log_intensity,intensity_ratio)
#print(priority)
#fuz.print_config(filename='pp')
#b=fuz.update_action()
#assert(np.linalg.norm(a-b)<0.1)
#print(fuz.config)
#fuz.fuzzy_ctrl.print_state()
