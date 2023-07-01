import os
from main import import_results

if __name__=="__main__":
    
    #ROBERTA CONFIG BRUTE FORCE
    TOP_K = [1, 3, 5, 7, 9, 10, 20, 30, 40, 50]
    WINDOW = [50, 100, 150, 200, 250, 500]
    STRIDE = [0, 50, 100]
    MODEL = 'roberta-base-squad-v2'
    # Run the pipeline for all combinations of split_size, split_overlap_size and top_k
    for w in WINDOW:
        for s in STRIDE:
            for k in TOP_K:
                if w > s:
                    _ = os.system(f'python3 main.py --split_size {w} --split_stride {s} --top_k {k}')
                    assert _==0, "RuntimeError: main.py did not run correctly"
    
    
    _ = os.system(f'python3 main.py --split_size {150} --split_stride {50} --top_k {5}')
    _ = os.system(f'python3 main.py --split_size {100} --split_stride {0} --top_k {5}')
    _ = os.system(f'python3 main.py --split_size {150} --split_stride {100} --top_k {10}')
                    
    # Create a summary of results and export them
    os.makedirs('./doc/summary/', exist_ok=True)
    
    # Import results
    import matplotlib.pyplot as plt
    results = import_results()
    IR_METRICS = ['recall_single_hit', 'recall_multi_hit', 'precision', 'map', 'mrr', 'ndcg']
    NLP_METRICS = ['rougeL', 'rouge1', 'rouge2', 'f1', 'exact_match', 'sas', 'meteor']
    RUNTIME_VARS = ['split_size', 'stride', 'top_k']
        
    # Check duplicates and select runtimevars
    duplicate = results['reader'][['runtime_hash','split_size', 'stride', 'top_k']].groupby(RUNTIME_VARS).count()
    duplicated = duplicate[duplicate['runtime_hash'] > 100]
    duplicated
    assert len(duplicated)==0, "The analysis contains double data points"
    results['reader'].loc[(results['reader']['stride']==50) &
                          (results['reader']['split_size']==150) & 
                          (results['reader']['top_k']==5)]['runtime_hash'].unique()
    
        # Retriever
    df_retriever = results['retriever'].loc[[tk in TOP_K for tk in results['retriever']['top_k']], :]
    retriever_out = df_retriever.select_dtypes(['float', 'int']).groupby(['split_size', 'stride', 'top_k']).mean().round(2)
    retriever_out.to_csv(os.path.join('./doc/summary/', 'results_retriever.csv'))
    print(retriever_out)
    
    ir_at_k = df_retriever.groupby('top_k')[IR_METRICS].mean().T.round(2)
    ir_at_k.index = ['$R_s$', '$R_m$', '$P$', '$MAP$', '$MRR$', '$NDCG$']
    ir_at_k.columns = ['@' + str(c) for c in ir_at_k.columns]
    ir_at_k.to_csv(os.path.join('./doc/summary/', 'results_retriever_@K.csv'))
    
    #P@K
    p_at_k = df_retriever.pivot_table(index=['split_size', 'stride'], columns='top_k', values='precision').round(2)
    p_at_k.columns = ['P@' + str(c) for c in p_at_k.columns]
    p_at_k.to_csv(os.path.join('./doc/summary/', 'results_retriever_P@K.csv'))
    p_at_k.mean().round(2).to_csv(os.path.join('./doc/summary/', 'results_retriever_P_m@K_mean.csv'))
    print(p_at_k)
    
    #R_s@K
    rs_at_k = df_retriever.pivot_table(index=['split_size', 'stride'], columns='top_k', values='recall_single_hit').round(2)
    rs_at_k.columns = ['R_s@' + str(c) for c in rs_at_k.columns]
    rs_at_k.to_csv(os.path.join('./doc/summary/', 'results_retriever_R_s@K.csv'))
    rs_at_k.mean().round(2).to_csv(os.path.join('./doc/summary/', 'results_retriever_R_s@K_mean.csv'))
    print(rs_at_k)
    
    #R_m@K
    rm_at_k = df_retriever.pivot_table(index=['split_size', 'stride'], columns='top_k', values='recall_multi_hit').round(2)
    rm_at_k.columns = ['R_m@' + str(c) for c in rm_at_k.columns]
    rm_at_k.to_csv(os.path.join('./doc/summary/', 'results_retriever_R_m@K.csv'))
    rm_at_k.mean().round(2).to_csv(os.path.join('./doc/summary/', 'results_retriever_R_m@K_mean.csv'))
    print(rm_at_k)
    
    # Plot results seperately using subplots
    n_row = len(df_retriever['split_size'].unique())
    n_col = len(df_retriever['stride'].unique())
    fig, ax = plt.subplots(n_row, n_col, figsize=(15, 15))
    for i, s in enumerate(sorted(df_retriever['split_size'].unique())):
        for j, o in enumerate(sorted(df_retriever['stride'].unique())):
            df = df_retriever.loc[(df_retriever['split_size'] == s) & (df_retriever['stride'] == o), :].sort_values('top_k')
            print(df)
            #df.plot(kind='line', x='top_k', y='precision', ax=ax[i, j], title=f"Split Size = {s}, Overlap = {o}", legend=False, ylim=(0, 1))
            df.plot(kind='line', x='top_k', y='recall_single_hit', ax=ax[i, j], title=f"Window Size = {s}, Stride = {o}", legend=False, ylim=(0.5, 1), color='orange')
            df.plot(kind='line', x='top_k', y='recall_multi_hit', ax=ax[i, j], title=f"Window Size = {s}, Stride = {o}", legend=False, ylim=(0.5, 1), color='red')
    plt.tight_layout()
    plt.xlabel()
    ax[0,2].legend(loc='upper right')
    plt.show()
    
    # Plot average metrics
    df_retriever[['top_k','recall_multi_hit', 'recall_single_hit', 'precision', 'map', 'mrr', 'ndcg']].groupby(['top_k']).mean().round(2).plot(kind='line', figsize=(10,5))
    plt.title(f"Average metrics for Top-k over all windows & strides")
    plt.xlabel('Top-k')
    plt.legend(['Recall Multi Hit', 'Recall Single Hit', 'Precision', 'MAP', 'MRR', 'NDCG'])
    plt.show()
    
        # Reader
    df_reader = results['reader']
    df_reader = df_reader.loc[df_reader['model'] == MODEL, :]

    analysis = df_reader.groupby(['runtime_hash', 'model'])[NLP_METRICS + RUNTIME_VARS].mean()
    #analysis.to_csv(os.path.join('./doc/summary/', 'results_roberta_hyperparameter_search.csv'))
    analysis.max().round(2) # Get best results
    analysis.corr().round(2).to_csv(os.path.join('./doc/summary/', 'results_correlation.csv')) # Correlation between metrics
    
    # Get best results
    df_out = df_reader.groupby(RUNTIME_VARS)[NLP_METRICS].mean().round(2).sort_values('rougeL', ascending=False)
    df_out.to_csv(os.path.join('./doc/summary/', 'results_roberta_hyperparameter_search.csv'))
    
    # Plot results for metrics
    for kind in ['line', 'bar']:
        # Split size
        ss = df_reader.select_dtypes(include=['float64', 'int']).groupby('split_size').mean().round(2)[NLP_METRICS]
        ss.plot(kind=kind, figsize=(10,5))
        plt.title(f"Window for {MODEL}")
        plt.show()
        # Overlap size
        os = df_reader.select_dtypes(include=['float64', 'int']).groupby('stride').mean().round(2)[NLP_METRICS]
        os.plot(kind=kind, figsize=(10,5))
        plt.title(f"Stride for {MODEL}")
        plt.show()
        # Top k
        tk = df_reader.select_dtypes(include=['float64', 'int']).groupby('top_k').mean().round(2)[NLP_METRICS]
        tk.plot(kind=kind, figsize=(10,5))
        plt.title(f"Top-k for {MODEL}")
        plt.show()
    
    # for split_size in sorted(df_reader['split_size'].unique()):
    #     df_s = df_reader.loc[df_reader['split_size'] == split_size, :]
    #     df_s.select_dtypes(include=['float64', 'int']).groupby('top_k').mean().round(2)[metrics].plot(kind='line', figsize=(10,5))
    #     plt.legend(loc='upper right')
    #     plt.title(f"Metrics for split_size={split_size}")
    #     plt.show()
    
    # botplot 
    tk.columns = ['ROUGE-L', 'ROUGE-1', 'ROUGE-2', 'F1', 'EM', 'SAS', 'METEOR']
    tk.plot(kind='box', figsize=(10,5))
    plt.title(f"Average metrics for Top-K for {MODEL} over all window and stride sizes")
    plt.show()
    
    
    
    
    
    
    # worst performing questions
    questions_results = results['reader'].groupby('question')[NLP_METRICS].mean()
    questions_results.sort_values('rougeL').head(10).index

