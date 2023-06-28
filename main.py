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
                         line=dict(color='white'), opacity=0.7, name='', showlegend=False))



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






# server = Flask(__name__)
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
# server = app.server






# Define the app layout
app.layout = html.Div([
    # Header
    dbc.Row([
        dbc.Col(html.H1("ADCC Data visualization", style={'textAlign': 'center', 'color': 'darkcyan', 'fontsize': '72px'}))
    ], style={'marginBottom': '40px', 'marginTop':'20px'}),
    
    html.Br(),
    
    # Content
    
    # 1st row title
    dbc.Row([html.H4(children='Match results over the years', style={'textAlign': 'center', 'marginBottom': '1px'})]),
    # 1st row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=targets_years), html.P("Arm submissions are generally becoming more rare since the 1st edition, when it was the most common way to end matches. With the exception of this edition in 1998, Neck attacks have always been the most likely submissions, having only tied with Leg attacks in 2011.",
                                                         style={'textAlign': 'center', 'max-width': '480px', 'marginLeft': '40px', 'marginRight':'10px'})],
                xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=methods_years), html.P(r"1998, 2005 and 2007 stand out as the most 'lethal' editions so far, with 52~53% of submissions. Decision victory had a surge from 2011 to 2015 and has since been stable at a considerable level. 2000 edition had more than 70% of its matches being decided by points.",
                                                        style={'textAlign': 'center', 'max-width': '480px', 'marginLeft': '70px', 'marginRight':'0px'})],
        xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0"),


    # 2nd row title
    dbc.Row([html.H4(children='Common performance metrics', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 2nd row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=win_sub_title), html.P("Only Jean Jacques Machado, Dean Lister and Gordon Ryan have been the top submission scorer in more than one ADCC edition. Dean Lister has the record at 3 editions. Meanwhile, Roger Gracie has the most submissions in a single edition at 8, followed by Marcelo Garcia at 7 in 2005 and 2007 respectively.",
                                                         style={'textAlign': 'center', 'max-width': '480px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=points), html.P("Heel hooks were the most popular submission in 2011 and 2013, both being editions when Dean Lister was the top submission artist. Despite that and two times Armbar held this position, RNC is by far the most recurrent popular submission.",
                                                style={'textAlign': 'center', 'max-width': '480px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0"),
    
    
    # 3rd row title
    dbc.Row([html.H4(children='Top submission and submission artists in each edition', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 3rd row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=sub_art), html.P("Marcelo Garcia, Gordon Ryan, Roger Gracie, Jean Jacques Machado and Dean Lister standing out, with both submission and win rates considerably higher than their peers'. Kade Ruotolo, Kron Gracie and Giancarlo Bodoni too, despite fewer total wins.",
                                                         style={'textAlign': 'center', 'max-width': '480px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=topsubs_year), html.P("Paulo Miyao, Royler Gracie and Ffion Davies stand out with great points differential (their total of points / their oponents')",
                                                style={'textAlign': 'center', 'max-width': '480px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0"),
    
    # 4th row title
    dbc.Row([html.H4(children='Biggest winners for each victory type', style={'textAlign': 'center', 'marginBottom': '1px', 'marginTop':'25px'})]),
    # 4th row content
    dbc.Row([
        dbc.Col([dcc.Graph(figure=type_frequents), html.P("""Again, Marcelo Garcia's career stands out for his ability to submit oponents in high level competition.  
                                                          Andre Galvao, on the other hand, seems like a more conservative player, knowing how to use the points system to win matches and achieve competition success.""",
                                                         style={'textAlign': 'center', 'max-width': '480px', 'marginLeft': '40px', 'marginRight':'10px'})],
        xs=7, sm=7, md=7, lg=5, xl=5),
        dbc.Col([dcc.Graph(figure=topwinner_targets), html.P("""Marcelo Garcia greatly stands out for the amount of matches ended by neck attack. Not only is he notorious for his use of guillotines, his seated guard systems for submission grappling are highly influential to this day.  

Dean Lister's known for introducing many of the leg attacks that later got popularized by teams focusing on the area, and has achieved many submissions with these attacks.  
  
Comparatively, arm attacks are rarer as a choice for specialization and its submissions seem to be more distributted among the athletes.""",
                                                style={'textAlign': 'center', 'max-width': '480px', 'marginLeft': '70px', 'marginRight':'0px'})],
                xs=7, sm=7, md=7, lg=5, xl=5)

    ], className="g-0"),
])



if __name__ == '__main__':
    app.run_server(debug=True)