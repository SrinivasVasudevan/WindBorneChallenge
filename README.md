# WindBorneChallenge

This is the repository that has the code used for my official submission to the Windborne challenge. First of all thanks to the Windborne team for this awesome challenge. The website should be live soon on an AWS instance.

The tech stack used: Flask (backend), React (Frontend), SQLite (DB Engine)

# Features
 - The balloons and their trajectories are shown in batches of 50 and the users of the webpage can click on next or prev to browse through the trajectories
 - Clicking a trajectory landmark (point used to build the trajectory) pulls up a small info about the trajectory
 - Clicking on the trajectory itself gives you an option to predict its next landmark (position in the next hour).

 # The challenge and my solution to it
 1. Query our **live constellation API**
    - The current positions of our global sounding balloons are at [`https://a.windbornesystems.com/treasure/00.json`](https://a.windbornesystems.com/treasure/00.json).
    - `01.json` is the state an hour ago, `03.json` is from three hours ago, etc, up to 23H ago.
    - The outputs are **undocumented** and may sometimes be **corrupted**—sorryy ¯\\_(ツ)_/¯.
    - You should robustly extract the available flight history for our constellation
    - ###Solution:
    - The backend first builds a history of the balloon's trajectory by parsing the response from the corresponding json files and storing it in the database.
    - Here there are several criteria that a new landmark point needs to satisfy to be part of the trajectory.
        - It needs to have travelled a distance/hour closer to the moving median of the past point
        - If it fails to do so, since the new landmark is known to be undocumented, it goes through another round of check that tries to associate this landmark with any of the landmarks of all other known trajectories.
        - If this check also fails, it gets marked as possibly corrupted, gets a prediction point for that timestamp, creates a new temporary trajectory to which it could be a part of (this is important because the new point we gathered could also be a new balloon getting released)
        - We assure robustness by introducing predictions to corrupted points by using a GaussianProcessRegressor to predict the corrupted point based on history with a liner extrapolation fallback. 

2. Find another existing **public dataset/API** and **combine** these two into something!
    - The world is your oyster—have fun with it 🤔
    - This should be the meat of the project, explore and find something **interesting and cool** to you!
    - What insights or problems could you tackle with both of these data feeds?
    - Remember, our API is **live**, so whatever you build should **update dynamically** with the latest `24H` of data.
    - Add a sentence to the `notes` explaining why you chose the external api/dataset you did!
    - ###Solution:
    - I got open-meteo's help here which is a free weather api. 
    - The sole reason this was added to introduce more robustness to real time prediction.
    - While using GaussianProcessRegressor to predict points gave decent predictions, using additional influencing factors would make the prediciton more robust.
    - Also to respect the fact that I am using a public api, I have put in a hard check on the number of api calls.
    - This also influenced my decision to use the wind data only in new predictions rather than in the corruption correction stage.

3. **Host your creation** on a publicly accessible URL
    - We can't wait to see what you build!
    - this should be the actual interative webpage, not a static repo
    - ###Solution:
    - Should be live through AWS.


## Plan of action
### Backend stuff
- [x] Read from the endpoint and segregate it into trajectory buckets
- [x] Take into account the possible data corruptions and smooth out the data (i am also gonna assume that data might sometime go poof)
- [x] I am assuming that the [x,y,z] provided as balloon coords are [lat,long,alt] but i can straight up change that later (this assumptions comes into play only when i bring in the public api to the picture)
- [x] I am going to be fancy and provide predictions for the trajectory by 
    - [x] taking into account the past trajectory
    - [x] Predictions with windspeed
- [x] Fetch data every 1hr to take into account the data refreshing.
- [x] Make it more fancy later, first get this done

>[!NOTICE]
> I have finished the backend except for the public repo/api and fetching every hour. 
> Fetching every hour can be based on my current function
> handle the public repo/api without hitting limits first

### Frontend stuff
- [x] Create a Mercator map background (using cesium here instead) (my lat,long assumptions is also going to play a decent role here)
- [x] Visualize trajectory by
    - [ ] Using google maps api (this potentially replaces the background overlay) and mark your coordinates
    - [ ] Use the image and do a lot of janky js stuff to plot the trajectory
- [ ] Show seperate marking for points where u are not confident about its trajectory
    #### Brownie Points - Frontend stuff
    - [x] Make the trajectories interactable






