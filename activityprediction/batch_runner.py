import logging
from surround import Runner
from stages import ActivitypredictionData
import pandas as pd

logging.basicConfig(level=logging.INFO)

class BatchRunner(Runner):
    def run(self, is_training=False):
        self.assembler.init_assembler(True)
        data = ActivitypredictionData()

        # Load data to be processed
        raw_data = "TODO: Load raw data here"

        # Setup input data
        data_one=pd.read_csv("data/1.csv",header=None)
        data_two=pd.read_csv("data/2.csv",header=None)
        data_three=pd.read_csv("data/3.csv",header=None)
        data_four=pd.read_csv("data/4.csv",header=None)
        data_five=pd.read_csv("data/5.csv",header=None)
        data_six=pd.read_csv("data/6.csv",header=None)
        data_seven=pd.read_csv("data/7.csv",header=None)
        data_eight=pd.read_csv("data/8.csv",header=None)
        data_nine=pd.read_csv("data/9.csv",header=None)
        data_ten=pd.read_csv("data/10.csv",header=None)
        data_eleven=pd.read_csv("data/11.csv",header=None)
        data_twelve=pd.read_csv("data/12.csv",header=None)
        data_thirteen=pd.read_csv("data/13.csv",header=None)
        data_fourteen=pd.read_csv("data/14.csv",header=None)
        data_fifteen=pd.read_csv("data/15.csv",header=None)

        df=pd.concat([data_one,data_two,data_three,data_four,data_five,data_six,data_seven,data_eight,data_nine,data_ten,data_eleven,data_twelve,data_thirteen,data_fourteen,data_fifteen],axis=0)
        df.columns=['seq_no','acc_x','acc_y','acc_z','activity']
        data.input_data=df
        # Run assembler
        self.assembler.run(data, is_training)

        logging.info("Batch Runner: %s", data.output_data)
