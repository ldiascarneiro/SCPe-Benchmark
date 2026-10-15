# Created by Leandro Carneiro at 5/11/25
# Description: 
# -----------------------------------------------------------
# main.py
from graph.laudo_graph import run_graph
from datetime import datetime
import config

if __name__ == "__main__":
    time_ini = datetime.now()

    run_graph(
        limit_top_sections=0,
        preview_only=False,
        start_row=config.DATASET_START_ROW,
        num_rows=config.DATASET_NUM_ROWS_TO_PROCESS,
    )

    time_end = datetime.now()
    print(f'Tempo de processamento: {time_end - time_ini}')
