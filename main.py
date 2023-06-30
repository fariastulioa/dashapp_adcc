import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash.dependencies import Input, Output, State
import numpy as np




# Set the default template to 'seaborn'
pio.templates.default = "seaborn"

# Get the 'seaborn' template
template = pio.templates[pio.templates.default]

# Make all backgrounds transparent
template.layout.paper_bgcolor = 'rgba(0,0,0,0)'
template.layout.plot_bgcolor = 'rgba(0,0,0,0)'

# Update the template in Plotly
pio.templates[pio.templates.default] = template

# Reading data
fdf = pd.read_csv('fighters_dataset.csv')
mdf = pd.read_csv('adcc_matches.csv')


# Plots the distribution between win rates and submission rates
# Generates the chart object by instanciating a plotly express histogram
win_sub_title = px.scatter(fdf, x='win_ratio', y='sub_win_ratio', hover_name="name",
                color=fdf['champion'].astype(str), labels={'color': 'Has title?'},
                size='total_wins', width=600, height=600)


win_sub_title.update_layout(
    title={
        'text': "Win ratio vs Submission ratio",
        'y':0.95,
        'x':0.48,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Win ratio", yaxis_title="Submission ratio", legend=dict(x=0.05, y=0.99,bgcolor='rgba(0, 0, 0, 0)')

)

# Edit legend labels for understandability
newnames = {'0':'No', '1': 'Title winner'}
win_sub_title.for_each_trace(lambda t: t.update(name = newnames[t.name]))

# Display observation under title explaining bubble size
legend_annotations = [
    dict(
        x=0.52,
        y=1,
        xref="paper",
        yref="paper",
        text=f"(Bubble size proportional to total wins)",
        showarrow=False,
        font=dict(size=12),
    )]
win_sub_title.update_layout(annotations=legend_annotations)

# Add a constant dashed line at the x=y diagonal
win_sub_title.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                        line=dict(color='lightgray'), opacity=0.7, name='', showlegend=False))



points = px.scatter(fdf, x='suffered_points_per_fight', y='scored_points_per_fight', hover_name="name",
                color=fdf['female'].astype(str), labels={'color': 'Sex'}, width=600, height=600)




points.update_layout(
    title={
        'text': "Average points Scored Vs. Average points Conceded",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Average points conceded per fight", yaxis_title="Average points scored per fight",
    legend=dict(x=0.75, y=0.99,bgcolor='rgba(0, 0, 0, 0)'), yaxis=dict(side='right')

)

# Renames the legend labels for understandability
newnames = {'0':'Male', '1': 'Female'}
points.for_each_trace(lambda t: t.update(name = newnames[t.name]))



# Generating methods over years plot
# Calculate likelihood for each category over the years and convert to percentages
likelihood_mdf = mdf.groupby(['year', 'victory_method']).size().div(mdf.groupby('year').size(), level='year').reset_index(name='likelihood')
likelihood_mdf['likelihood'] *= 100

# Create stacked bar chart using plotly express
methods_years = px.bar(likelihood_mdf, x='year', y='likelihood', color='victory_method',
            labels={'year': 'Year', 'likelihood': 'Probability (%)', 'victory_method': 'Victory Method'},
            title='Proportion of Victory Methods over the Years',
            hover_data={'year': True, 'likelihood': ':.2f', 'victory_method': True},
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Bold, width=600, height=600
            )

methods_years.update_layout(title_x=0.49, yaxis=dict(side='right'),legend_orientation='h',
                                legend=dict(x=0.02, y=1.1,bgcolor='rgba(0, 0, 0, 0)'),
                                legend_title_text="")


# Generating target over the years plot

# Calculate likelihood for each category over the years and convert to percentages
likelihood_mdf = mdf.groupby(['year', 'submission_target']).size().div(mdf.groupby('year').size(), level='year').reset_index(name='likelihood')
likelihood_mdf['likelihood'] *= 100

# Create line chart using plotly express
targets_years = px.line(likelihood_mdf, x='year', y='likelihood', color='submission_target',
            hover_name='submission_target', width=600, height=600,
            color_discrete_sequence=px.colors.qualitative.Dark24)

targets_years.update_traces(line_width=3)


targets_years.update_yaxes(title_text='Submission ratio (%)')
targets_years.update_xaxes(title_text='Year')

targets_years.update_layout(
    legend=dict(x=0.69, y=0.99,bgcolor='rgba(0, 0, 0, 0)'),
    legend_title_text="Target", title_x=0.5,
    title_text='Submission ratio for each target body part over the years',
    title_y=0.936
)


# Plotting top submission artists 
# Get only matches that ended by submission
submissions_mdf = mdf[mdf['victory_method'] == 'SUBMISSION']


# Group data to calculate the count of submissions for each fighter and year
grouped_mdf = submissions_mdf.groupby(['year', 'winner_name']).size().reset_index(name='count')

# Find the fighter with the most submission wins in each year
max_submissions = grouped_mdf.groupby('year').apply(lambda x: x.loc[x['count'].idxmax()]).reset_index(drop=True)

# Plot the horizontal bars
sub_art = go.Figure(data=go.Bar(
    y=max_submissions['year'],
    x=max_submissions['count'],
    text=max_submissions['winner_name'],
    hovertemplate=
    '<b>Year</b>: %{y}<br>' +
    '<b>Fighter</b>: %{text}<br>' +
    '<b>Submissions</b>: %{x}<extra></extra>',
    orientation='h',
    textposition='inside',  # Set text position inside the bars
    textfont={'size': 14},  # Adjust the text font size
    marker_color='steelblue',  # Customize the bar color
))


sub_art.update_layout(
    title='Fighter with Most Submission Wins in Each Year',
    xaxis_title='Number of Submissions',
    yaxis_title='Year',
    height = 800,
    title_x=0.48,
)

# Plotting top submission for each year

# Group data to calculate the count of submissions for each fighter and year
grouped_mdf = submissions_mdf.groupby(['year', 'submission']).size().reset_index(name='count')

# Find the fighter with the most submission wins in each year
max_submissions = grouped_mdf.groupby('year').apply(lambda x: x.loc[x['count'].idxmax()]).reset_index(drop=True)

# Plot the horizontal bars
topsubs_year = go.Figure(data=go.Bar(
    y=max_submissions['year'],
    x=max_submissions['count'],
    text=max_submissions['submission'],
    hovertemplate=
    '<b>Year</b>: %{y}<br>' +
    '<b>Fighter</b>: %{text}<br>' +
    '<b>Submissions</b>: %{x}<extra></extra>',
    orientation='h',
    textposition='inside',  # Set text position inside the bars
    textfont={'size': 14},  # Adjust the text font size
    marker_color='steelblue',  # Customize the bar color
))


topsubs_year.update_layout(
    title='Most frequent submission in each year',
    xaxis_title='Number of Submissions',
    yaxis_title='Year',
    height = 800,
    title_x=0.48,
    yaxis=dict(
        title='Year',
        side='right',  # Move the title to the right side
        position=1,  # Adjust the position of the title
    )
)



# Plotting most frequent winners and losers for each target
# Get counts of submissions by fighter for each target
winner_counts = mdf.groupby(['submission_target', 'winner_name']).size().reset_index(name='win_count')

# Find the most frequent winner for each submission target
top_winners = winner_counts.groupby('submission_target')['win_count'].idxmax()
most_frequent_winners = winner_counts.loc[top_winners]

# Group the data by submission target and loser name to get the count of losses for each combination
loser_counts = mdf.groupby(['submission_target', 'loser_name']).size().reset_index(name='loss_count')

# Find the most frequent loser for each submission target
top_losers = loser_counts.groupby('submission_target')['loss_count'].idxmax()
most_frequent_losers = loser_counts.loc[top_losers]

# Create a horizontal bar chart using Plotly
topwinner_targets = go.Figure()

# Add green bars for the most frequent winners
topwinner_targets.add_trace(go.Bar(
    y=most_frequent_winners['submission_target'],
    x=most_frequent_winners['win_count'],
    orientation='h',
    name='Most Frequent Winner',
    marker=dict(color='seagreen'),
    text=most_frequent_winners['winner_name'],
    textposition='inside', opacity=0.8,
    textfont=dict(color='white', size=14)
))

# Add red bars for the most frequent losers
topwinner_targets.add_trace(go.Bar(
    y=most_frequent_losers['submission_target'],
    x=most_frequent_losers['loss_count'],
    orientation='h',
    name='Most Frequent Loser',
    marker=dict(color='orangered'),
    text=most_frequent_losers['loser_name'],
    textposition='inside', opacity=0.8,
    textfont=dict(color='white', size=14)
))


topwinner_targets.update_layout(
    title='Most Frequent Winner and Loser for Each Submission Target',
    xaxis_title='Number of matches',
    yaxis_title='Submission Target',
    barmode='relative',
    bargap=0.2,
    legend=dict(
        x=0.45,
        y=0.97,
        bgcolor='rgba(0,0,0,0)'
    ),
    showlegend=True,
    title_x=0.45, width=600, height=600,
        yaxis=dict(
        title='Submission target',
        side='right',  # Move the title to the right side
        position=1,  # Adjust the position of the title
    )
)


# Plotting most frequent winner by each victory type (and loser)

# Exclude minor exceptions from the analysis
victories_mdf = mdf[(mdf['victory_method'] != 'INJURY') & (mdf['victory_method'] != 'DESQUALIFICATION')].copy()

# Group the data by submission target and winner name to get the count of wins for each combination
winner_counts = victories_mdf.groupby(['victory_method', 'winner_name']).size().reset_index(name='win_count')

# Find the most frequent winner for each submission target
top_winners = winner_counts.groupby('victory_method')['win_count'].idxmax()
most_frequent_winners = winner_counts.loc[top_winners]

# Group the data by submission target and loser name to get the count of losses for each combination
loser_counts = victories_mdf.groupby(['victory_method', 'loser_name']).size().reset_index(name='loss_count')

# Find the most frequent loser for each submission target
top_losers = loser_counts.groupby('victory_method')['loss_count'].idxmax()
most_frequent_losers = loser_counts.loc[top_losers]

# Create a horizontal bar chart using Plotly
type_frequents = go.Figure()

# Add green bars for the most frequent winners
type_frequents.add_trace(go.Bar(
    y=most_frequent_winners['victory_method'],
    x=most_frequent_winners['win_count'],
    orientation='h',
    name='Most Frequent Winner',
    marker=dict(color='seagreen'),
    text=most_frequent_winners['winner_name'],
    textposition='inside', opacity=0.8,
    textfont=dict(color='white', size=16)
))

# Add red bars for the most frequent losers
type_frequents.add_trace(go.Bar(
    y=most_frequent_losers['victory_method'],
    x=most_frequent_losers['loss_count'],
    orientation='h',
    name='Most Frequent Loser',
    marker=dict(color='orangered'),
    text=most_frequent_losers['loser_name'],
    textposition='inside', opacity=0.8,
    textfont=dict(color='white', size=16)
))

# Set the layout
type_frequents.update_layout(
    title='Most frequent Winner and Loser of each Victory method',
    xaxis_title='Number of matches',
    yaxis_title='Vcitory by',
    barmode='relative',
    bargap=0.2,
    legend=dict(
        x=0.7,
        y=0.02,
        bgcolor='rgba(0,0,0,0)'
    ),
    showlegend=True,
    title_x=0.47, width=600, height=600
)




# Get only matches that ended in submission
submission_mdf = mdf[mdf['victory_method'] == 'SUBMISSION']

# Calculate submission ratio for each 'importance' value
submission_ratio = submission_mdf.groupby('importance').size() / mdf.groupby('importance').size()
submission_ratio *= 100  # Convert to percentage

# Create a DataFrame with 'importance' and 'submission_ratio' columns
submission_ratio_mdf = pd.DataFrame({'importance': submission_ratio.index, 'submission_ratio': submission_ratio.values})

# Plot bars
sub_imp = px.bar(data_frame=submission_ratio_mdf, x=submission_ratio_mdf['importance'].astype(str),
            y='submission_ratio',labels={'importance': 'Importance',
                                        'submission_ratio': 'Submission Ratio (%)',
            'x':'Match importance'},
            title='Probability of Submission by match importance',
            text=submission_ratio.values.round(2),  # Display percentage values inside bars
            hover_data={'importance': True, 'submission_ratio': ':.2f'},
            color_continuous_scale=px.colors.sequential.Sunsetdark,
            color='importance', hover_name='importance')


# Modify the tick names in the x-axis
custom_labels = {
'0':'Other',
'1':'Quarterfinals',
'2':'Semifinals',
'3':'3rd place',
'4':'Finals',
'5':'Superfight'
}

sub_imp.update_xaxes(ticktext=list(custom_labels.values()),
                tickvals=list(custom_labels.keys()))

sub_imp.update_traces(textposition='inside')  # Position the text inside the bars
sub_imp.update_layout(showlegend=False, coloraxis_showscale=False,
                title_x=0.48)



# Get only matches the ended in decision
decision_mdf = mdf[mdf['victory_method'] == 'DECISION']

# Calculate decision ratio for each 'importance' value
decision_ratio = decision_mdf.groupby('importance').size() / mdf.groupby('importance').size()
decision_ratio *= 100  # Convert to percentage

# It's easier to plot this exact information by creating a specific dataframe
decision_ratio_mdf = pd.DataFrame({'importance': decision_ratio.index, 'decision_ratio': decision_ratio.values})

# Plot the bars with plotly express
dec_imp = px.bar(data_frame=decision_ratio_mdf, x=decision_ratio_mdf['importance'].astype(str),
            y='decision_ratio',labels={'importance': 'Importance',
                                        'Decision victory ratio (%)': 'decision Ratio (%)',
        'x':'Match importance'},
            title='Probability of Decision by match importance',
            text=decision_ratio.values.round(2),  # Display percentage values inside bars
            hover_data={'importance': True, 'decision_ratio': ':.2f'},
            color_continuous_scale=px.colors.sequential.Sunsetdark,
            color='importance')


dec_imp.update_xaxes(ticktext=list(custom_labels.values()),
                tickvals=list(custom_labels.keys()))

dec_imp.update_traces(textposition='inside') 
dec_imp.update_layout(showlegend=False, coloraxis_showscale=False,
                yaxis_title="Probability of victory by decision (%)",
                title_x=0.48, yaxis=dict(side='right'))



# Group the data by 'weight_class' and 'submission_target' and calculate the relative frequency
grouped_data = mdf.groupby(['weight_class', 'submission_target']).size().reset_index(name='count')
grouped_data['relative_frequency'] = grouped_data.groupby('weight_class')['count'].transform(lambda x: x / x.sum())

# Create the bar chart using Plotly Express
weight_target = px.bar(grouped_data, x='weight_class', y='relative_frequency',
            color='submission_target')

# Calculate the x-coordinate for each annotation
grouped_data['cumulative_relative_frequency'] = grouped_data.groupby('weight_class')['relative_frequency'].cumsum() - 0.5 * grouped_data['relative_frequency']
grouped_data['x_annotation'] = grouped_data['weight_class']

# Add annotations to display the submission target names
for _, row in grouped_data.iterrows():
    weight_target.add_annotation(
        x=row['x_annotation'],
        y=row['cumulative_relative_frequency'],
        text=row['submission_target'],
        showarrow=False,
        font=dict(color='white', size=12),
        textangle=0,
        xanchor='center',
        yanchor='middle'
    )

weight_target.update_traces(hovertemplate='%{x} Weight Class: %{y} submission ratio')

weight_target.update_layout(
    title='Target body part of successful submissions',
    xaxis_title='Weight class',
    yaxis_title='Ratio to total submissions in that weight class',
    barmode='relative',
    bargap=0.2,
    showlegend=False,
    title_x=0.5, yaxis=dict(side='right')
)

# Modify the tick names in the x-axis
custom_labels = {
'0':'66kg (60kg for females)',
'1':'77kg',
'2':'88kg',
'3':'99kg',
'4':'+99kg (+60kg for females)',
}

weight_target.update_xaxes(ticktext=list(custom_labels.values()),
                tickvals=list(custom_labels.keys()))





total_counts = mdf.groupby('weight_class').size().reset_index(name='total_count')
submission_counts = mdf[mdf['victory_method'] == 'SUBMISSION'].groupby('weight_class').size().reset_index(name='submission_count')
probability_data = pd.merge(total_counts, submission_counts, on='weight_class', how='left')
probability_data['probability'] = (probability_data['submission_count'] / probability_data['total_count']) * 100

# Create the bar chart using Plotly Express
weight_sub = px.bar(probability_data, x='weight_class', y='probability',
            color='weight_class', color_continuous_scale=px.colors.sequential.Sunsetdark)

# Add static text annotations of probability values inside each bar
for i, row in probability_data.iterrows():
    weight_sub.add_annotation(
        x=row['weight_class'],
        y=row['probability']-1.2,
        text=f'<b>{np.round(row["probability"], 2)}%</b>',
        showarrow=False,
        font=dict(color='lightsteelblue', size=14),
        textangle=0,
        xanchor='center',
        yanchor='middle'
    )

weight_sub.update_traces(hovertemplate='%{x} Weight Class: %{y:.2f}% submission probability')
weight_sub.update_layout(
    title='Probability of victory by any submission in different weight classes',
    xaxis_title='Weight class',
    yaxis_title='Submission probability (%)',
)

# Modify the tick names in the x-axis
custom_labels = {
'0':'66kg (60kg for females)',
'1':'77kg',
'2':'88kg',
'3':'99kg',
'4':'+99kg (+60kg for females)',
}

weight_sub.update_xaxes(ticktext=list(custom_labels.values()),
                tickvals=list(custom_labels.keys()))
weight_sub.update_layout(showlegend=False, coloraxis_showscale=False,title_x=0.5)




# Group the data by 'weight_class' and 'submission_target' and calculate the relative frequency
grouped_data = mdf.groupby(['absolute', 'submission_target']).size().reset_index(name='count')
grouped_data['relative_frequency'] = grouped_data.groupby('absolute')['count'].transform(lambda x: x / x.sum())

# Create the bar chart using Plotly Express
open_sub = px.bar(grouped_data, x='absolute', y='relative_frequency',
             color='submission_target')

# Calculate the x-coordinate for each annotation
grouped_data['cumulative_relative_frequency'] = grouped_data.groupby('absolute')['relative_frequency'].cumsum() - 0.5 * grouped_data['relative_frequency']
grouped_data['x_annotation'] = grouped_data['absolute']

# Add annotations to display the submission target names
for _, row in grouped_data.iterrows():
    open_sub.add_annotation(
        x=row['x_annotation'],
        y=row['cumulative_relative_frequency'],
        text=row['submission_target'],
        showarrow=False,
        font=dict(color='white', size=12),
        textangle=0,
        xanchor='center',
        yanchor='middle'
    )


open_sub.update_layout(
    title='Submission target',
    xaxis_title='',
    yaxis_title='Ratio to total submissions in match type',
    barmode='relative',
    bargap=0.2,
    showlegend=False, yaxis=dict(side='right')
)

open_sub.update_layout(
    title_x=0.5,
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 1],
        ticktext = ['Weight class divisions', 'Open weight (absolute) division']
    )
)




likelihood_mdf = mdf.groupby(['absolute', 'victory_method']).size().div(mdf.groupby('absolute').size(), level='absolute').reset_index(name='likelihood')
likelihood_mdf['likelihood'] *= 100

# Create stacked bar chart using plotly express
open_vic = px.bar(likelihood_mdf, x='absolute', y='likelihood', color='victory_method',
            labels={'likelihood': 'Probability (%)', 'victory_method': 'Victory by'},
            title='Victory types on Open weight Vs. Weight class divisions',
            hover_data={'absolute': True, 'likelihood': ':.2f', 'victory_method': True},
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.T10
            )

open_vic.update_layout(
    title_x=0.45,
    xaxis_title='',
    xaxis = dict(
        tickmode = 'array',
        tickvals = [0, 1],
        ticktext = ['Weight class divisions', 'Open weight (absolute) division']
    )
)



# Plots the distribution between win rates and submission rates
# For athletes who competed in finals or superfights
highlvl = fdf[fdf['highest_match_importance'] > 3].copy()
highlvl.sort_values(by='n_titles', inplace=True, ascending=False)

# Generates the chart object by instantiating a plotly express scatter plot
high_winsub = px.scatter(highlvl, x='win_ratio', y='sub_win_ratio', hover_name="name",
                color=highlvl['n_titles'].astype(str),
                size='total_wins', size_max=40, opacity=0.6,
                color_discrete_sequence=px.colors.sequential.Plasma,
                labels={'color': 'Number of titles'},
                width=640, height=640)

# Add a constant dashed line at the x=y diagonal
high_winsub.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                         line=dict(color='white'), opacity=0.7, name=''))

high_winsub.update_layout(
    title={
        'text': "Win ratio vs Submission ratio",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Win ratio", yaxis_title="Submission ratio"
)

# Display observation under title explaining bubble size
legend_annotations = [
    dict(
        x=0.54,
        y=0.99,
        xref="paper",
        yref="paper",
        text=f"(Bubble size proportional to total wins)",
        showarrow=False,
        font=dict(size=12),
    )
]

high_winsub.update_layout(annotations=legend_annotations)






# Generates the chart object by instanciating a plotly express histogram
high_subscore = px.scatter(highlvl, x='custom_score', y='sub_win_ratio', hover_name="name",
                 color='favorite_target', labels={'favorite_target': 'Most frequent submission target'},
                width=720, height=640, size='total_wins', log_x=True,
                color_discrete_sequence=px.colors.qualitative.Bold, size_max=40)




high_subscore.update_layout(
    title={
        'text': "Submission ratio by custom score",
        'y':0.95,
        'x':0.42,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Custom score", yaxis_title="Submission ratio"

)


# Display observation under title explaining bubble size
legend_annotations = [
    dict(
        x=0.52,
        y=0.99,
        xref="paper",
        yref="paper",
        text=f"(Bubble size proportional to Titles won)",
        showarrow=False,
        font=dict(size=12),
    )]


high_subscore.update_layout(annotations=legend_annotations, yaxis=dict(side='right'))





# Generates the chart object by instanciating a plotly express histogram 
highest_match = px.histogram(x=fdf['highest_match_importance'])
highest_match.update_layout(
    title={
        'text': "Distribution of highest match importance fighter competed in",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Match importance",
    yaxis_title="Number of athletes"
)

# Calculates the percentage for each value
total_count = len(fdf['highest_match_importance'])
percentages = fdf['highest_match_importance'].value_counts(normalize=True) * 100

# Adds the percentage text on top of each bar
highest_match.update_traces(text=percentages.round(2).astype(str) + '%', textposition='auto')

# Modify the tick names in the x-axis
custom_labels = {
'0':'Other',
'1':'Quarterfinals',
'2':'Semifinals',
'3':'3rd place',
'4':'Finals',
'5':'Superfight'
}

highest_match.update_xaxes(ticktext=list(custom_labels.values()),
                tickvals=list(custom_labels.keys()))





# Generates the chart object by instanciating a plotly express histogram 
total_wins = px.histogram(fdf, x='total_wins')
total_wins.update_layout(
    title={
        'text': "Distribution of total matches the athlete won",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Total matches won by athlete",
    yaxis_title="Number of athletes", yaxis=dict(side='right')
)

# Calculates the percentage for each value
total_count = len(fdf['total_wins'])
percentages = fdf['total_wins'].value_counts(normalize=True) * 100

# Add the percentage text on top of each bar
total_wins.update_traces(text=percentages.round(2).astype(str) + '%', textposition='auto')


targets = pd.DataFrame({
    'most_vulnerable': ['No specific target', 'Neck', 'Leg', 'Arm', 'Other/Unknown'],
    'count_most_vulnerable': [293, 162, 70, 70, 19],
    'favorite_target': ['No specific target', 'Neck', 'Leg', 'Arm', 'Other/Unknown'],
    'count_favorite_target': [491, 64, 33, 22, 4]
})

body_parts = go.Figure()

# Add green bars for count of favorite_target
body_parts.add_trace(go.Bar(
    x=targets['favorite_target'],
    y=targets['count_favorite_target'],
    name='Favorite Target',
    marker_color='lightseagreen'
))

# Add red bars for count of most_vulnerable
body_parts.add_trace(go.Bar(
    x=targets['most_vulnerable'],
    y=targets['count_most_vulnerable'],
    name='Most Vulnerable Target',
    marker_color='indianred'
))

# Set the layout of the bar chart
body_parts.update_layout(
    xaxis_title='Category',
    yaxis_title='Count',
    title='Counts of Most Vulnerable and Favorite Target Categories',
    barmode='group',
    legend=dict(
        orientation="v",
        yanchor="bottom",
        y=0.84,
        xanchor="right",
        x=1,
        bgcolor='rgba(0, 0, 0, 0)'
    )
)

body_parts.update_layout(legend=dict(
    orientation="v",
    yanchor="bottom",
    y=0.84,
    xanchor="right",
    x=1,
    bgcolor='rgba(0, 0, 0, 0)'
),
    xaxis_title="Submission target body part", yaxis_title="Number of athletes",
    title_text='Most frequent body part in submissions on and from each athlete',
    title_x=0.47
)



# Generates the chart object by instanciating a plotly express histogram 
n_subs = px.histogram(fdf, x='n_different_subs', hover_name="n_different_subs")
n_subs.update_layout(
    title={
        'text': "Distribution of number of different submissions",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Amount of different submissions scored by the athlete",
    yaxis_title="Number of athletes", yaxis=dict(side='right')
)

# Calculates the percentage for each value
total_count = len(fdf['n_different_subs'])
percentages = fdf['n_different_subs'].value_counts(normalize=True) * 100

# Add the percentage text on top of each bar
n_subs.update_traces(text=percentages.round(2).astype(str) + '%', textposition='auto')




winrate_debut = px.scatter(fdf, x='debut_year', y='win_ratio', hover_name="name",
                color=fdf['favorite_target'].astype(str),
                size='total_wins', size_max=30, opacity=0.6, labels={'color': 'Favorite target'})


winrate_debut.update_layout(
    title={
        'text': "Win rate by debut year",
        'y':0.95,
        'x':0.47,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Year of first appearance", yaxis_title="Win rate"

)

# Display observation under title explaining bubble size
legend_annotations = [
    dict(
        x=0.53,
        y=1.04,
        xref="paper",
        yref="paper",
        text=f"(Bubble size proportional to total wins)",
        showarrow=False,
        font=dict(size=12),
    )]

winrate_debut.update_layout(annotations=legend_annotations)






# Generates the chart object by instanciating a plotly express histogram
subs_debut = px.scatter(fdf, x='debut_year', y='sub_win_ratio', hover_name="name",
                 color=fdf['favorite_target'].astype(str),
                size='total_wins', size_max=30, opacity=0.6, labels={'color': 'Favorite target'})


subs_debut.update_layout(
    title={
        'text': "Submission rate by debut year",
        'y':0.95,
        'x':0.47,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Year of first appearance", yaxis_title="Submission rate"

)

# Display observation under title explaining bubble size
legend_annotations = [
    dict(
        x=0.53,
        y=1.04,
        xref="paper",
        yref="paper",
        text=f"(Bubble size proportional to total wins)",
        showarrow=False,
        font=dict(size=12),
    )]

subs_debut.update_layout(annotations=legend_annotations)




fdf.sort_values(by='n_titles', ascending=True, inplace=True)
# Generates the chart object by instanciating a plotly express histogram
titles= px.scatter(fdf, x='custom_score', y='sub_win_ratio', hover_name="name",
                 color=fdf['n_titles'].astype(str), labels={'color': 'Number of titles'},
                width=720, height=640, size='total_wins', log_x=True,
                color_discrete_sequence=px.colors.sequential.Sunsetdark, size_max=40)


titles.update_layout(
    title={
        'text': "Submission ratio by custom score",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Custom score", yaxis_title="Submission ratio"

)


# Display observation under title explaining bubble size
legend_annotations = [
    dict(
        x=0.542,
        y=1,
        xref="paper",
        yref="paper",
        text=f"(Bubble size proportional to total wins)",
        showarrow=False,
        font=dict(size=12),
    )]

titles.update_layout(annotations=legend_annotations)





# Generates the chart object by instanciating a plotly express histogram
win_imp = px.scatter(fdf, x='avg_match_importance', y='win_ratio', hover_name="name",
                 color=fdf['champion'].astype(str), labels={'color': 'Has title?'},
                size='total_wins', size_max=30,
                width=640, height=640)


win_imp.update_layout(
    title={
        'text': "Win ratio by average match importance for fighters",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Average match importance", yaxis_title="Win ratio"

)

# Display observation under title explaining bubble size
legend_annotations = [
    dict(
        x=0.546,
        y=1,
        xref="paper",
        yref="paper",
        text=f"(Bubble size proportional to total wins)",
        showarrow=False,
        font=dict(size=12),
    )]
win_imp.update_layout(annotations=legend_annotations, yaxis=dict(side='right'))


# Renames the legend labels for understandability
newnames = {'0':'No', '1': 'Title winner'}
win_imp.for_each_trace(lambda t: t.update(name = newnames[t.name]))




# Get only matches that ended by heel hook
heel_hook_mdf = mdf[mdf['submission'].isin(['Inside heel hook', 'Heel hook', 'Outside heel hook'])]

# Group the DataFrame to calculate the count of submissions by year and submission type
grouped_mdf = heel_hook_mdf.groupby(['year']).size().reset_index(name='count')

# Create the plot using Plotly Express
heelhook = px.bar(grouped_mdf, x='year', y='count',
             labels={'year': 'Year', 'count': 'Number of Heel Hooks'},
             title='Number of Heel Hook submissions over the Years',
             hover_data={'year': True, 'count': True})

heelhook.update_layout(title_x=0.49)




# Get only matches that ended by knee bar
kb_mdf = mdf[mdf['submission'] == 'Kneebar']

# Group the DataFrame to calculate the count of submissions by year and submission type
grouped_mdf = kb_mdf.groupby(['year']).size().reset_index(name='count')

# Create the plot using Plotly Express
kneebar = px.bar(grouped_mdf, x='year', y='count',
             labels={'year': 'Year', 'count': 'Number of Kneebars'},
             title='Number of Kneebar submissions over the Years',
             hover_data={'year': True, 'count': True})

kneebar.update_layout(title_x=0.49, yaxis=dict(side='right'))




sunburst_df = mdf.copy()

sunburst_df['submission_target'] = sunburst_df['submission_target'].apply(lambda x: x if x else ' ')
sunburst_df['submission_target'].fillna(' ', inplace=True)
sunburst_df['submission'] = sunburst_df['submission'].apply(lambda x: x if x else ' ')
sunburst_df['submission'].fillna(' ', inplace=True)

tmap = px.treemap(
    sunburst_df,
    path=['victory_method', 'submission_target', 'submission'],
    values=sunburst_df.index,
    hover_data=['victory_method', 'submission_target', 'submission'],
    color_discrete_map=px.colors.qualitative.Dark24
)

tmap.update_traces(
    hovertemplate='<b>%{label}</b><br>Percentage of section: %{percentParent:.2%}',
    textinfo='label+percent root'
)

tmap.update_layout(
    title='"Win by": Victory method distribution (hover to see local percentages)',
    treemapcolorway=px.colors.qualitative.Dark24,
)


import plotly.subplots as sp

# Female
sdf_sex1 = mdf[mdf['female'] == 1]
counts_sex1 = sdf_sex1['victory_method'].value_counts()

# Male
sdf_sex0 = mdf[mdf['female'] == 0]
counts_sex0 = sdf_sex0['victory_method'].value_counts()

sexpies = sp.make_subplots(rows=1, cols=2, subplot_titles=('Female', 'Male'), specs=[[{'type':'pie'}, {'type':'pie'}]])

palette_sex1 = px.colors.qualitative.Pastel

sexpies.add_trace(go.Pie(labels=counts_sex1.index, values=counts_sex1,
                     marker=dict(colors=palette_sex1), textinfo='label+percent', hovertemplate='%{label}: %{percent}<extra>Female</extra>'), row=1, col=1)

palette_sex0 = px.colors.qualitative.Dark2

sexpies.add_trace(go.Pie(labels=counts_sex0.index, values=counts_sex0,
                     marker=dict(colors=palette_sex0), textinfo='label+percent', hovertemplate='%{label}: %{percent}<extra>Male</extra>'), row=1, col=2)

sexpies.update_layout(showlegend=False, title='Victory Method Distribution by Sex')








app = dash.Dash(__name__,
                external_stylesheets=[{
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
            dbc.themes.BOOTSTRAP,
            dbc.icons.BOOTSTRAP]) #server = server
server = app.server


@app.callback(
    Output('output-bar-chart', 'figure'),
    Input('year-slider', 'value')
)
def update_bar_chart(year_range):
    min_year, max_year = year_range
    filtered_df = mdf[(mdf['year'] >= min_year) & (mdf['year'] <= max_year)]
    top_athletes = filtered_df['winner_name'].value_counts().nlargest(5)

    fig = px.bar(
        top_athletes,
        x=top_athletes.index,
        y=top_athletes.values,
        labels={'x': 'Athlete', 'y': 'Number of Wins'},
        title=f'Top 5 Athletes with Most Wins (From {min_year} to {max_year})',
        color_continuous_scale=px.colors.sequential.Teal,
        color = top_athletes.values,
    )

    fig.update_layout(xaxis_title="Athlete name", showlegend=False, coloraxis_showscale=False)
    
    return fig




@app.callback(
    Output('output-sub-chart', 'figure'),
    Input('year-slider', 'value')
)
def update_sub_chart(year_range):
    min_year, max_year = year_range
    filtered_df = mdf[(mdf['year'] >= min_year) & (mdf['year'] <= max_year)]
    filtered_df = filtered_df[filtered_df['victory_method'] == 'SUBMISSION']
    top_athletes = filtered_df['winner_name'].value_counts().nlargest(5)

    fig = px.bar(
        top_athletes,
        x=top_athletes.index,
        y=top_athletes.values,
        labels={'x': 'Athlete', 'y': 'Number of Submissions'},
        title=f'Top 5 Athletes with Most Submissions (From {min_year} to {max_year})',
        color_continuous_scale=px.colors.sequential.Teal,
        color = top_athletes.values,
    )

    fig.update_layout(xaxis_title="Athlete name", showlegend=False, coloraxis_showscale=False)
    
    return fig


titles_df = mdf[mdf['importance'] > 3]


@app.callback(
    Output('output-title-chart', 'figure'),
    Input('year-slider', 'value')
)
def update_title_chart(year_range):
    min_year, max_year = year_range
    filtered_df = titles_df[(titles_df['year'] >= min_year) & (titles_df['year'] <= max_year)]
    top_athletes = filtered_df['winner_name'].value_counts().nlargest(5)

    fig = px.bar(
        top_athletes,
        x=top_athletes.index,
        y=top_athletes.values,
        labels={'x': 'Athlete', 'y': 'Number of Titles'},
        title=f'Top 5 Athletes with Title wins (From {min_year} to {max_year})',
        color_continuous_scale=px.colors.sequential.Teal,
        color = top_athletes.values,
    )

    fig.update_layout(xaxis_title="Athlete name", showlegend=False, coloraxis_showscale=False)
    
    return fig



# Define the app layout
app.layout = html.Div([
    # Header
    #dbc.Row([
    #    dbc.Col(html.H1("ADCC Data visualization", style={'textAlign': 'center', 'color': 'darkcyan', 'fontsize': '72px'}))
    #], style={'marginBottom': '40px', 'marginTop':'20px'}),
    html.Img(src='/assets/banner.png', style={'width': '100%', 'marginBottom':'36px'}),
    
    html.Br(),
    
    html.Div(
    [
        dcc.Markdown(
            """
            This is a data visualization dashboard for analyzing the ADCC submission grappling competition.  
            It is based on data scraped from BJJ Heroes website and
            used to create two datasets: one containing information on individual matches and one focused on individual athletes.  
              
            The purpose of this dashboard is to make certain trends and relationships from the data visible and understandable and promote objective insights
            for fans of the sport.  
             
            Sports media and fans discussion tend to be very passionate, with narratives often stemming from subjective perceptions,
            emotional reactions and personal preferences.  
            In this context, data analysis and visualization tools can lead to more significant and accurate narratives and enrich insights or disprove common misconceptions.  
            
            For more information, accessing the code or downloading the datasets to use as you wish, visit [Kaggle](https://www.kaggle.com/code/albucathecoder/adcc-fighters-eda-and-clustering-kmeans)  
              
                
            
            Author:  
            [GitHub](https://github.com/fariastulioa)  
            [LinkedIn](https://www.linkedin.com/in/tuliofarias/)
            """
        ),
    ],
    style={'textAlign': 'center', 'textJustify': 'inter-word', 'padding': '20px'},
    ),
    
    
    
    # Content
    
    
        
    # 1st row title
    dbc.Row([html.H4(children='Match results over the years', style={'textAlign': 'center', 'marginBottom': '1px'})]),
    # 1st row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=targets_years), html.Div([dcc.Markdown("""
Arm submissions are generally becoming more rare since the 1st edition, when it was the most common way to end matches.  
With the exception of this edition in 1998, Neck attacks have always been the most likely submissions, having only tied with Leg attacks in 2011.
"""),],style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'},)],
                xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=methods_years), html.Div([dcc.Markdown("""
1998, 2005 and 2007 stand out as the most 'lethal' editions so far, with 52~53% of submissions.  
Decision victory had a surge from 2011 to 2015 and has since been stable at a considerable level.  
2000 edition had more than 70% of its matches being decided by points.
""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
        xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0",  justify="evenly"),

    
    
    # 2nd row title
    dbc.Row([html.H4(children='Top submission and submission artists in each edition', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 2nd row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=sub_art), html.Div([dcc.Markdown("""Marcelo Garcia, Gordon Ryan, Roger Gracie, Jean Jacques Machado and Dean Lister standing out, with both submission and win rates considerably higher than their peers'.  
                                                                   Kade Ruotolo, Kron Gracie and Giancarlo Bodoni too, despite fewer total wins.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=topsubs_year), html.P("Paulo Miyao, Royler Gracie and Ffion Davies stand out with great points differential (their total of points / their oponents')",
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0",  justify="evenly"),
    
    # 3rd row title
    dbc.Row([html.H4(children='Biggest winners for each victory type', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 3rd row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=type_frequents), html.Div([dcc.Markdown("""Again, Marcelo Garcia's career stands out for his ability to submit oponents in high level competition.  
                                                                          Andre Galvao, on the other hand, seems like a more conservative player, knowing how to use the points system to win matches and achieve competition success.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=topwinner_targets), html.Div([dcc.Markdown("""Marcelo Garcia with a great number of matches ended by neck attack. Not only is he notorious for his use of guillotines, his seated guard systems for submission grappling are highly influential to this day.  
                                                                             Dean Lister's known for introducing many of the leg attacks that later got popularized by teams focusing on the area, and has achieved many submissions with these attacks.  
                                                                             Comparatively, arm attacks are rarer as a choice for specialization and its submissions seem to be more distributted among the athletes.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0",  justify="evenly"),
    
    # 4th row title
    dbc.Row([html.H4(children='Match outcome and importance relationship', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 4th row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=sub_imp), html.Div([dcc.Markdown("""The chart above clearly shows that the first round has the most risk of submissions.  
                                                                   The skill variance in this stage is the greatest, since all athletes are pooled together initially, which can explain the higher probability of submissions occuring.  
                                                                   This is reduced in later stages, where high skill gaps are less likely.  
                                                                   Superfights, on the other hand, have considerably lower submission rates.
                                                                     
                                                                       
                                                                   These matches happen only between the most skilled competitors, who have the most to lose. Not only are more skilled competitors more difficult to submit, they generally tend to play more defensively in these situations, which might explain these lower submission rates.  
                                                                     
                                                                     
                                                                   Semifinals display lower-than-expected submission rates, which might indicate athletes are more cautious at this stage, being faced with the prospect of competing in a final or losing the chance to do so.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=dec_imp), html.Div([dcc.Markdown("""The same trends can be seen here, with semifinals and superfights boasting high decision victory rates.  
                                                                   This means that not only are athletes more cautious about getting submitted, but also about conceding points in these situations.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0",  justify="evenly"),
    
    # 5th row title
    dbc.Row([html.H4(children='Match outcome and weight class relationship', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 5th row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=weight_sub), html.Div([dcc.Markdown("""There are no substantial differences that can be seen from the graph above that would define a specific trend in how submission rates behave in relation to weight in general, but the heaviest weight class has lower submission rates, as does the 88 kg one.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=weight_target), html.Div([dcc.Markdown("""As shown in the chart above, neck submissions are less common in matches between heavier athletes.  
                                                         Since these submissions are more likely to happen from dominant positions while leg and arm attacks are available from a wider range of situations, one can assume this trend is due to the fact that stronger, larger fighters are less likely to be controlled.  
                                                         Legs and arms are more exposed than the neck in the majority of positions, so athletes can target them without requiring to first secure a more dominant position.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0",  justify="evenly"),
    
    
    # 6th row title
    dbc.Row([html.H4(children='Open Weight (Absolute) vs Weight Class divisions', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 6th row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=open_vic), html.Div([dcc.Markdown("""In terms of victory method, on the other hand, the absolute division shows extremely similar results as the weight class ones.  
                                                                    One small yet significant difference that can be observed is in the probabilities of injury and disqualification, both more likely to happen in open weight matches.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=open_sub), html.Div([dcc.Markdown("""It can be seen above that arm and neck attacks are less common in matches in the absolute division.  
                                                                    This most likely reflects a trend of more frequent leg exchanges and entanglements in these matches, which are commonly pursued as an equalizing tactic by smaller fighters against larger ones.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0", justify="evenly"),
    
    
    # 7th row title
    dbc.Row([html.H4(children='Common performance metrics', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 7th row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=win_sub_title), html.Div([dcc.Markdown("""Only Jean Jacques Machado, Dean Lister and Gordon Ryan have been the top submission scorer in more than one ADCC edition. Dean Lister has the record at 3 editions.  
                                                                         Meanwhile, Roger Gracie has the most submissions in a single edition at 8, followed by Marcelo Garcia at 7 in 2005 and 2007 respectively.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=points), html.Div([dcc.Markdown("""Heel hooks were the most popular submission in 2011 and 2013, both being editions when Dean Lister was the top submission artist.  
                                                                  Despite that and two times Armbar held this position, RNC is by far the most recurrent popular submission.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0",  justify="evenly"),
    
    # 8th row title
    dbc.Row([html.H4(children='Stats distribution for high level athletes', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    dbc.Row([html.H6(children='(athletes that have competed in finals or superfights)', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'5px'})]),
    # 8th row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=high_winsub), html.Div([dcc.Markdown("""This type of chart showscases how impressive some athletes' achievements are, such as Ricardo Arona's undefeated status with 13 total wins and 4 titles.  
                                                                       The promise Kade Ruotolo showcased winning 4 matches by submission in his first appearance also stands out.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=high_subscore), html.Div([dcc.Markdown("""Custom Score is an engineered feature calculated using different metrics such as victories/losses at each competition level and number of appearances by the athlete.  
                                                                         This chart clearly displays the difference between those who never got to showcase their preferred target for submissions at the highest level and those who could.  
                                                                         It's also possible to see that Arm specialist in general displayed similar results with this metrics, so Fabricio Werdum, Ronaldo Souza, Alexandre Ribeiro and FFion Davies are packed in close proximity. Kade Ruotolo's an evident outlier from this group, as is Ricardo Arona for the Leg attackers.  
                                                                         Being the most common preferred target among athletes, Neck attackers are a more broadly distributted group here.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0",  justify="evenly"),
    
    # 9th row title
    dbc.Row([html.H4(children='Distribution of competition success among athletes', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 9th row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=highest_match), html.Div([dcc.Markdown("""It makes sense to logically expect this chart to have decreasing heights from left to right on the bars, since more important bouts are fought by athletes who bested others in previous rounds, but that's not what can be seen here.  
                                                                         Since there's a lot of missing data in the original data source (BJJ Heroes ADCC bouts stats), this distribution might be explained by the probable trend that more important bouts are more likely to have data available on the website.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=total_wins), html.Div([dcc.Markdown("""The distribution closely follows the logic that success in competitive sports can be visualized as a pyramid with the most successful athletes being a few at the top and many at the bottom with less success.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0",  justify="evenly"),
    
    # 10th row title
    dbc.Row([html.H4(children='Different submissions scored by athletes', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 10th row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=body_parts), html.Div([dcc.Markdown("""We can see that most fighters don't have enough submission data (from ADCC boughts, at least) available for us to determine what their preferred target when submitting oponents is or where they are most vulnerable to being submitted.  
                                                                      Still, it can be seen that there are more leg specialists than arm specialists but even combined they are not as numerous as the athletes who get their most submissions by neck attacks.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=n_subs), html.Div([dcc.Markdown("""The 0 bar shows that almost 80% of ADCC competitors has never submitted an opponent in the event.  
                                                                  Aproximately 10% have submission win(s) from a single submission and only the remaining 10% have won matches with different submissions.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5)
        

    ], className="g-0",  justify="evenly"),
    
    
    # 11st row title
    dbc.Row([html.H4(children='Athletes stats by debut year', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 11st row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=winrate_debut), html.Div([dcc.Markdown("""It's worth noting that a "talent drought" or shortage of capable fresh athletes was never experienced in ADCC, with fighters from all the different generations going on to achieve impressive and or unprecedented feats.  
                                                                         Alexandre Ribeiro had the most total wins among athletes who had their debut in the first decade (1998-2008) of the competition, Andre Galvao in the second decade (2008-2018)and Gordon Ryan in the third and current decade (2018-present).
""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=subs_debut), html.Div([dcc.Markdown("""This plot highlights the impressive achievements of different athletes than the previous ones, since there's a comparison being made within "generations".  
                                                                      Examples are Jean Jaques Machado, Dean Lister, Kade Ruotolo, Kron Gracie and Rousimar Palhares.  
                                                                      Bianca Mesquita and Ana Carolina Vieira are also considerably ahead of their peers.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5)
        

    ], className="g-0",  justify="evenly"),
    
    # 12nd row title
    dbc.Row([html.H4(children='Further explorations on titles distribution', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 12nd row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=titles), html.Div([dcc.Markdown("""The plot above highlights some of the most memorable athletes who were never champions such as Rousimar Palhares, Craig Jones, Joao Miyao and Ricardo Almeida.  
                                                                  It also makes some exceptional champions stand out such as Gordon Ryan, Marcelo Garcia and Ricardo Arona.""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=win_imp), html.Div([dcc.Markdown("""It can be seen that the average match importance for fighters is not a good feature to differentiate them, since most of them appear closely packed together in the above chart.  
                                                                   The trend observed here (of positive correlation between match importance and average win rate) is to be expected since fighters advance to later stages (matches with higher importance) by winning matches with less importance.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5)
        

    ], className="g-0",  justify="evenly"),
    
    
    # 13rd row title
    dbc.Row([html.H4(children='Changes in leg attacks over the years', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 13rd row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=heelhook), html.Div([dcc.Markdown("""Heel hooks have transitioned from a rare occurance to one of the most used leg attacks in submission grappling, with a surge in 2011 followed by its popularization among the athlete pool.
""")],
                                                style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=kneebar), html.Div([dcc.Markdown("""Kneebars, on the other hand, are becoming more rare since 2001. After the rise of the heel hook in 2011, it has only happened twice.""")],
                                                        style={'textAlign': 'center', 'max-width': '640px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5)
        

    ], className="g-0",  justify="evenly"),
    
    
    # 14th row title
    dbc.Row([html.H4(children='Overal distribution of how matches ended', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 14th row content
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Graph(figure=tmap, style={'textAlign': 'center'}),
                    html.Div([dcc.Markdown(
                        """
                        Since athletes do not wear kimonos for ADCC matches, It's surprising to find that armbars are 
                        so prevalent among arm targetting attacks, considering the grips for shoulder locks such as 
                        kimuras and americanas are usually harder to get out of in no-gi grappling.  
                                                
                        Leg attacks are the group with the most variety and more evenly spread numbers, which can be 
                        attributed to the fact that leg entries and entanglements often threaten multiple lower body 
                        submissions simultaneously. In this context, it makes sense for athletes to try for a variety 
                        of these finishing moves when going for these positions.  
                                                
                        It's worth noting that the overall submission rates for ADCC are good from a spectator's 
                        perspective, as the constant threat of match-ending action make watching them more enticing.
                        """)],
                        style={'textAlign': 'center', 'marginLeft': '50px', 'marginRight':'50px', 'marginTop':'0px'}
                    )
                ],
                width=12,
                className='justify-content-center align-items-center', xs=9, sm=9, md=9, lg=7, xl=7
            )
        ],
        className='g-0 justify-content-center align-items-center'
    ),
    
    
    # 15th row title
    dbc.Row([html.H4(children='Match outcome and athletes sex', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 15th row content
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Graph(figure=sexpies, style={'textAlign': 'center', 'marginBottom':'1px', 'marginLeft':'auto', 'marginRight':'auto'}),
                    html.Div([dcc.Markdown(
                        """
                        Women have been thrice as likely to have matches ended due to injury than men, and also have considerable more judges decisions to define match outcome.  
                        The submission rates are surprisingly similar, despite common notion that women in general are more technically responsible about defense than men.
                        """)],
                        style={'textAlign': 'center', 'marginLeft': '50px', 'marginRight':'50px', 'marginTop':'0px'}
                    )
                ],
                width=12,
                className='justify-content-center align-items-center', xs=9, sm=9, md=9, lg=7, xl=7
            )
        ],
        className='g-0 justify-content-center align-items-center'
    ),
    
    dbc.Row(html.Hr(style={'border-color': 'lightgray', 'marginBottom':'0px', 'marginTop':'25px'})),
    # 0th row title
    dbc.Row([html.H3(children='Visualizing specific time frames', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'4px', 'color': 'darkcyan', 'fontWeight': 'bold'})]),
    dbc.Row([html.H6(children='(Drag start and end year to specify a time interval)', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'5px', 'color': 'lightskyblue'})]),
    # 0h row content
    dbc.Row(
        [
            dbc.Col(
                [   
                    html.Div([dcc.Graph(id='output-bar-chart'),
                    
])
                ],
                width=12,
                className='justify-content-center align-items-center', xs=9, sm=9, md=9, lg=7, xl=7
            ),dbc.Col([html.Div([dcc.Markdown(
                        """
                        These charts help better understand and contextualize all the *"of all time"* discussions that happens through the years.  
                        The chronological navigation can put the total bulk of accomplishments in perspective for athletes of each generation.  
                        It can also to some degree be used to gauge the longevity of some elite grapplers competitiveness.
                        """)],
                        style={'textAlign': 'center', 'marginLeft': '50px', 'marginRight':'50px', 'marginTop':'20px'}
                    ),])
            
        ],
        className='g-0 justify-content-center align-items-center', style={'marginBottom': '1px', 'marginTop':'1px'}
    ),
    dbc.Row([dcc.RangeSlider(
                        id='year-slider',
                        min=mdf['year'].min(),
                        max=mdf['year'].max(),
                        value=[mdf['year'].min(), mdf['year'].max()],
                        marks={str(year): str(year) for year in mdf['year'].unique()},
                        step=None
                        )],style={'textAlign': 'center', 'marginLeft': '50px', 'marginRight':'50px', 'marginTop':'5px', 'marginBottom':'5px'}),
    dbc.Row([html.H6(children='(Drag years on the slider above)', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'5px', 'color': 'lightskyblue'})]),
    dbc.Row([
        dbc.Col(html.Div([dcc.Graph(id='output-sub-chart')])),
        dbc.Col(html.Div([dcc.Graph(id='output-title-chart')]))

    ], className="g-0",  justify="evenly"),
    
])



app.title = 'ADCC Data'


if __name__ == '__main__':
    app.run_server(debug=True)