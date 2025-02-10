# WindBorneChallenge

## Plan of action
### Backend stuff
- [ ] Read from the endpoint and segregate it into trajectory buckets
- [ ] Take into account the possible data corruptions and smooth out the data (i am also gonna assume that data might sometime go poof)
- [ ] I am assuming that the [x,y,z] provided as balloon coords are [lat,long,alt] but i can straight up change that later (this assumptions comes into play only when i bring in the public api to the picture)
- [] I am going to be fancy and provide predictions for the trajectory by 
    - [ ] taking into account the past trajectory
    - [ ] Predictions with windspeed
- [ ] Fetch data every 1hr to take into account the data refreshing.
- [ ] Make it more fancy later, first get this done

### Frontend stuff
- [ ] Create a Mercator map background (my lat,long assumptions is also going to play a decent role here)
- [ ] Visualize trajectory by
    - [ ] Using google maps api (this potentially replaces the background overlay) and mark your coordinates
    - [ ] Use the image and do a lot of janky js stuff to plot the trajectory
- [ ] Show seperate marking for points where u are not confident about its trajectory
    #### Brownie Points - Frontend stuff
    - [ ] Make the trajectories interactable






