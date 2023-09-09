import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dython.nominal import _REPLACE, _DEFAULT_REPLACE_VALUE, _comp_assoc
from sklearn.decomposition import PCA
from copy import deepcopy


### FUNCTIONS TAKEN FROM TABLEEVALUATOR
# monkey patching from dython
def associations(dataset,
                 nominal_columns='auto',
                 mark_columns=False,
                 theil_u=False,
                 plot=True,
                 clustering=False,
                 bias_correction=True,
                 nan_strategy=_REPLACE,
                 nan_replace_value=_DEFAULT_REPLACE_VALUE,
                 ax=None,
                 figsize=None,
                 annot=True,
                 fmt='.2f',
                 cmap=None,
                 sv_color='silver',
                 cbar=True
                 ):
    """
    Calculate the correlation/strength-of-association of features in data-set
    with both categorical and continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Cramer's V or Theil's U for categorical-categorical cases

    Parameters:
    -----------
    dataset : NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    nominal_columns : string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can
        also be the string 'all' to state that all columns are categorical,
        'auto' (default) to try to identify nominal columns, or None to state
        none are categorical
    mark_columns : Boolean, default = False
        if True, output's columns' names will have a suffix of '(nom)' or
        '(con)' based on there type (eda_tools or continuous), as provided
        by nominal_columns
    theil_u : Boolean, default = False
        In the case of categorical-categorical feaures, use Theil's U instead
        of Cramer's V
    plot : Boolean, default = True
        Plot a heat-map of the correlation matrix
    clustering : Boolean, default = False
        If True, hierarchical clustering is applied in order to sort
        features into meaningful groups
    bias_correction : Boolean, default = True
        Use bias correction for Cramer's V from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328.
    nan_strategy : string, default = 'replace'
        How to handle missing values: can be either 'drop_samples' to remove
        samples with missing values, 'drop_features' to remove features
        (columns) with missing values, or 'replace' to replace all missing
        values with the nan_replace_value. Missing values are None and np.nan.
    nan_replace_value : any, default = 0.0
        The value used to replace missing values with. Only applicable when
        nan_strategy is set to 'replace'
    ax : matplotlib ax, default = None
        Matplotlib Axis on which the heat-map will be plotted
    figsize : (int,int) or None, default = None
        a Matplotlib figure-size tuple. If `None`, falls back to Matplotlib's
        default. Only used if `ax=None`.
    annot : Boolean, default = True
        Plot number annotations on the heat-map
    fmt : string, default = '.2f'
        String formatting of annotations
    cmap : Matplotlib colormap or None, default = None
        A colormap to be used for the heat-map. If None, falls back to Seaborn's
        heat-map default
    sv_color : string, default = 'silver'
        A Matplotlib color. The color to be used when displaying single-value
        features over the heat-map
    cbar: Boolean, default = True
        Display heat-map's color-bar

    Returns:
    --------
    A dictionary with the following keys:
    - `corr`: A DataFrame of the correlation/strength-of-association between
    all features
    - `ax`: A Matplotlib `Axe`

    Example:
    --------
    See examples under `dython.examples`
    """
    corr, columns, nominal_columns, inf_nan, single_value_columns = _comp_assoc(dataset.copy(deep=True), nominal_columns, mark_columns,
                                                                                theil_u, clustering, bias_correction,
                                                                                nan_strategy, nan_replace_value)
    for col in corr.columns:
        corr[col] = corr[col].astype(float)

    if ax is None:
        plt.figure(figsize=figsize)
    if inf_nan.any(axis=None):
        inf_nan_mask = np.vectorize(lambda x: not bool(x))(inf_nan.values)
        ax = sns.heatmap(inf_nan_mask,
                         cmap=['white'],
                         annot=inf_nan if annot else None,
                         fmt='',
                         center=0,
                         square=True,
                         ax=ax,
                         mask=inf_nan_mask,
                         cbar=False)
    else:
        inf_nan_mask = np.ones_like(corr)
    if len(single_value_columns) > 0:
        sv = pd.DataFrame(data=np.zeros_like(corr),
                          columns=columns,
                          index=columns)
        for c in single_value_columns:
            sv.loc[:, c] = ' '
            sv.loc[c, :] = ' '
            sv.loc[c, c] = 'SV'
        sv_mask = np.vectorize(lambda x: not bool(x))(sv.values)
        ax = sns.heatmap(sv_mask,
                         cmap=[sv_color],
                         annot=sv if annot else None,
                         fmt='',
                         center=0,
                         square=True,
                         ax=ax,
                         mask=sv_mask,
                         cbar=False)
    else:
        sv_mask = np.ones_like(corr)
    mask = np.vectorize(lambda x: not bool(x))(inf_nan_mask) + np.vectorize(lambda x: not bool(x))(sv_mask)
    ax = sns.heatmap(corr,
                     cmap=cmap,
                     annot=annot,
                     fmt=fmt,
                     center=0,
                     vmax=1.0,
                     vmin=-1.0 if len(columns) - len(nominal_columns) >= 2 else 0.0,
                     square=True,
                     mask=mask,
                     ax=ax,
                     cbar=cbar)
    if plot:
        plt.show()
    return {'corr': corr,
            'ax': ax}


def cdf(data_r, data_f, xlabel: str = 'Values', ylabel: str = 'Cumulative Sum', ax=None):
    """
    Plot continous density function on optionally given ax. If no ax, cdf is plotted and shown.

    :param data_r: Series with real data
    :param data_f: Series with fake data
    :param xlabel: Label to put on the x-axis
    :param ylabel: Label to put on the y-axis
    :param ax: The axis to plot on. If ax=None, a new figure is created.
    """
    x1 = np.sort(data_r)
    x2 = np.sort(data_f)
    y = np.arange(1, len(data_r) + 1) / len(data_r)

    ax = ax if ax else plt.subplots()[1]

    axis_font = {'size': '14'}
    ax.set_xlabel(xlabel, **axis_font)
    ax.set_ylabel(ylabel, **axis_font)

    ax.grid()
    ax.plot(x1, y, marker='o', linestyle='none', label='Real', ms=8)
    ax.plot(x2, y, marker='o', linestyle='none', label='Fake', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    import matplotlib.ticker as mticker

    # If labels are strings, rotate them vertical
    if isinstance(data_r, pd.Series) and data_r.dtypes == 'object':
        ticks_loc = ax.get_xticks()
        dif = len(ticks_loc) - len(data_r.sort_values().unique())
        tick_labels = [val for val in data_r.sort_values().unique()]
        for _ in range(dif):
            tick_labels.append('')
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(tick_labels, rotation='vertical')

    if ax is None:
        plt.show()


def plot_cumsums(real, fake, nr_cols=4, fname=None):
    """
    Plot the cumulative sums for all columns in the real and fake dataset. Height of each row scales with the length of the labels. Each plot contains the
    values of a real columns and the corresponding fake column.

    :param real: The real dataset as a DataFrame.
    :param fake: The fake dataset as a DataFrame.
    :param nr_cols: The number of columns in the plotted figure. Defaults to 4.
    :param fname: If not None, saves the plot with this file name.
    """
    nr_charts = len(real.columns)
    nr_rows = max(1, nr_charts // nr_cols)
    nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

    max_len = 0
    # Increase the length of plots if the labels are long
    if not real.select_dtypes(include=['object']).empty:
        lengths = []
        for d in real.select_dtypes(include=['object']):
            lengths.append(max([len(x.strip()) for x in real[d].unique().tolist()]))
        max_len = max(lengths)

    row_height = 6 + (max_len // 30)
    fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(16, row_height * nr_rows))
    fig.suptitle('Cumulative Sums per feature', fontsize=16)
    axes = ax.flatten()
    for i, col in enumerate(real.columns):
        r = real[col]
        f = fake.iloc[:, real.columns.tolist().index(col)]
        cdf(r, f, col, 'Cumsum', ax=axes[i])
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    if fname is not None:
        plt.savefig(fname)


def plot_correlation_difference(real: pd.DataFrame, fake: pd.DataFrame, plot_diff: bool = True, cat_cols: list = None,
                                annot=False, fname=None):
    """
    Plot the association matrices for the `real` dataframe, `fake` dataframe and plot the difference between them. Has support for continuous and Categorical
    (Male, Female) data types. All Object and Category dtypes are considered to be Categorical columns if `dis_cols` is not passed.
    - Continuous - Continuous: Uses Pearson's correlation coefficient
    - Continuous - Categorical: Uses so called correlation ratio (https://en.wikipedia.org/wiki/Correlation_ratio) for both continuous - categorical and categorical - continuous.
    - Categorical - Categorical: Uses Theil's U, an asymmetric correlation metric for Categorical associations

    :param real: DataFrame with real data
    :param fake: DataFrame with synthetic data
    :param plot_diff: Plot difference if True, else not
    :param cat_cols: List of Categorical columns
    :param boolean annot: Whether to annotate the plot with numbers indicating the associations.
    """
    assert isinstance(real, pd.DataFrame), f'`real` parameters must be a Pandas DataFrame'
    assert isinstance(fake, pd.DataFrame), f'`fake` parameters must be a Pandas DataFrame'
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    if cat_cols is None:
        cat_cols = real.select_dtypes(['object', 'category'])
    if plot_diff:
        fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    real_corr = associations(real, nominal_columns=cat_cols, plot=False, theil_u=True,
                             mark_columns=True, annot=annot, ax=ax[0], cmap=cmap)['corr']
    fake_corr = associations(fake, nominal_columns=cat_cols, plot=False, theil_u=True,
                             mark_columns=True, annot=annot, ax=ax[1], cmap=cmap)['corr']

    if plot_diff:
        diff = abs(real_corr - fake_corr)
        sns.set(style="white")
        sns.heatmap(diff, ax=ax[2], cmap=cmap, vmax=.3, square=True, annot=annot, center=0,
                    linewidths=.5, cbar_kws={"shrink": .5}, fmt='.2f')

    titles = ['Real', 'Fake', 'Difference'] if plot_diff else ['Real', 'Fake']
    for i, label in enumerate(titles):
        title_font = {'size': '18'}
        ax[i].set_title(label, **title_font)
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)


def convert_numerical(real, fake, categorical_columns):
    """
    Special function to convert dataset to a numerical representations while making sure they have identical columns. This is sometimes a problem with
    categorical columns with many values or very unbalanced values

    :param real: The original 'real' dataset.
    :param fake: The 'fake' dataset to be matched to the 'real' one.
    :param categorical_columns: List of columns that are to be converted to numerical format.

    :return: Real and fake dataframe factorized using the pandas function.
    """
    real = real
    fake = fake
    for c in categorical_columns:
        if real[c].dtype == 'object':
            real[c] = pd.factorize(real[c], sort=True)[0]
            fake[c] = pd.factorize(fake[c], sort=True)[0]

    return real, fake


def plot_pca(real_, fake_, categorical_columns, fname=None):
    """
    Plot the first two components of a PCA of real and fake data.

    :param real_: The original 'real' dataset.
    :param fake_: The 'fake' dataset to be compared with the 'real' one.
    :param categorical_columns: List of columns that are to be converted to numerical format before applying PCA.
    :param fname: If not none, saves the plot with this file name.
    """
    real = deepcopy(real_)
    fake = deepcopy(fake_)
    real, fake = convert_numerical(real, fake, categorical_columns)

    pca_r = PCA(n_components=2)
    pca_f = PCA(n_components=2)

    real_t = pca_r.fit_transform(real)
    fake_t = pca_f.fit_transform(fake)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('First two components of PCA', fontsize=16)
    sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
    sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
    ax[0].set_title('Real data')
    ax[1].set_title('Fake data')

    if fname is not None:
        plt.savefig(fname)
