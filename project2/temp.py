def graph_performance_with_k(k, SSE, BSS, sil, distance_measure=1):
    plt.rcParams['figure.dpi'] = 150
    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(9,3)
    # fig.tight_layout()
    axes[0].plot(k, SSE, color=purples[2], label='SSE(cohe)')
    axes[0].plot(k, BSS, color=purples[5], label='BSS(sep)')
    axes[0].set_title('k vs. SSE and BSS (Distance {})'.format(str(distance_measure)))
    axes[0].set_ylabel('SSE and BSS')
    axes[0].set_xlabel('k')
    axes[0].legend()

    # silhouette plotting
    axes[1].plot(k, sil, color=purples[4])
    axes[1].set_title('k vs. Silhouette Score (Distance {})'.format(str(distance_measure)))
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_xlabel('k')
    plt.show();