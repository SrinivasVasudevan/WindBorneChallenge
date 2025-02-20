import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { Ion, Viewer, Cartesian3, Color, Entity, ScreenSpaceEventHandler, ScreenSpaceEventType, defined, PolylineGlowMaterialProperty, PolylineGraphics } from "cesium";
import { Button, Dialog, DialogTitle, DialogContent, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, List, ListItem, ListItemButton, ListItemText, Divider, IconButton } from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
Ion.defaultAccessToken = "token_goes_here";

function App() {
  const cesiumContainerRef = useRef(null);
  const viewerRef = useRef(null);
  const [trajectories, setTrajectories] = useState([]);
  const [selectedBalloon, setSelectedBalloon] = useState(null);
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [page, setPage] = useState(0);
  const itemsPerPage = 50;
  const [showingPageOnly, setShowingPageOnly] = useState(true);

  const balloonIds = Array.from(new Set(trajectories.map(t => t.balloon_id)));
  const totalPages = Math.ceil(balloonIds.length / itemsPerPage);
  const displayedBalloonIds = balloonIds.slice(page * itemsPerPage, (page + 1) * itemsPerPage);
  const visibleBalloonIds = showingPageOnly ? displayedBalloonIds : balloonIds;
  const [predictionData, setPredictionData] = useState(null);

  useEffect(() => {
    if (cesiumContainerRef.current && !viewerRef.current) {
      viewerRef.current = new Viewer(cesiumContainerRef.current, {
        shouldAnimate: true,
      });
    }
    return () => {
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!viewerRef.current) return;
    const fetchTrajectory = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5000/api/trajectories");
        const data = await response.json();
        setTrajectories(data);
      } catch (error) {
        console.error("Error fetching trajectory:", error);
      }
    };
    fetchTrajectory();
    const interval = setInterval(fetchTrajectory, 3600000);
    //return () => clearInterval(interval);
  }, [viewerRef.current]);

  useEffect(() => {
    if (!viewerRef.current || !trajectories.length) return;
    const viewer = viewerRef.current;
    viewer.entities.removeAll();
    
    const filteredTrajectories = selectedBalloon 
      ? trajectories.filter(t => t.balloon_id === selectedBalloon) 
      : trajectories.filter(t => displayedBalloonIds.includes(t.balloon_id));
    
    filteredTrajectories.forEach((trajectory) => {
    
      
      if (trajectory.points && trajectory.points.length > 1) {
        const isSelected = selectedBalloon === trajectory.balloon_id;
        const positions = trajectory.points
          .filter(point => 
            typeof point.lng === "number" &&
            typeof point.lat === "number" &&
            typeof point.altitude === "number" &&
            point.predicted === false && 
            point.corrupted === 0
          )
          .map(point => Cartesian3.fromDegrees(point.lng, point.lat, point.altitude * (isSelected ? 55000 : 50000)));
        
        
        
        if (positions.length > 1) {
          const ent = viewer.entities.add({
            name: `Trajectory ${trajectory.balloon_id}`,
            polyline: new PolylineGraphics({
              positions: positions,
              width: 2,
              followSurface: true,
              material: new PolylineGlowMaterialProperty({
                glowPower: 0.2,
                taperPower: 0.5,
                color: isSelected ? Color.YELLOW : Color.CORNFLOWERBLUE,
              }),
              clampToGround: false
            })
          });
          if (isSelected) {
            viewer.flyTo(ent, {
              duration: 1.5
            });
          }

        }

        trajectory.points.forEach((point, index) => {
          let pointColor = Color.YELLOW;
          if (point.predicted) {
            pointColor = Color.GREEN;
          } else if (point.corrupted) {
            pointColor = Color.BLUE;
          }

          const entity = viewer.entities.add({
            position: Cartesian3.fromDegrees(point.lng, point.lat, point.altitude * (isSelected ? 55000 : 50000)),
            point: {
              pixelSize: 5,
              color: pointColor,
            },
            description: JSON.stringify(point, null, 2)
          });
          entity.pointData = point;
        });

        if(predictionData){
            predictionData.forEach((point) => {
              viewer.entities.add({
                position: Cartesian3.fromDegrees(point.lng, point.lat, point.altitude * 50000),
                point: {
                  pixelSize: 5,
                  color: Color.RED,
                },
                description: `Predicted Point - Lat: ${point.lat}, Lng: ${point.lng}, Alt: ${point.altitude}`,
              });
              
            });
        }
        
      }
    });

    const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((movement) => {
      const pickedObject = viewer.scene.pick(movement.position);
      if (defined(pickedObject) && pickedObject.id && pickedObject.id.pointData) {
        setSelectedPoint(pickedObject.id.pointData);
      }
      if (defined(pickedObject) && pickedObject.id && pickedObject.id.name) {
        setSelectedBalloon(pickedObject.id.name.replace("Trajectory ", ""));
      }
    }, ScreenSpaceEventType.LEFT_CLICK);
  }, [trajectories, selectedBalloon, page]);

  const handlePredict = async () => {
    if (!selectedBalloon) return;
    try {
      const response = await fetch(`http://127.0.0.1:5000/api/predict/${selectedBalloon}`);
      const prediction = await response.json();
      setPredictionData(prediction);
      if (viewerRef.current) {
        const viewer = viewerRef.current;
        prediction.forEach((point) => {
          const ent = viewer.entities.add({
            position: Cartesian3.fromDegrees(point.lng, point.lat, point.altitude * 50000),
            point: {
              pixelSize: 5,
              color: Color.RED,
            },
            description: `Predicted Point - Lat: ${point.lat}, Lng: ${point.lng}, Alt: ${point.altitude}`,
          });
          viewer.flyTo(ent, {
            duration: 1.5
          });
        });
      }
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={{ width: "300px", background: "#222", color: "white", padding: "10px", overflowY: "auto", borderRight: "2px solid #444" }}>
        <h3 style={{ textAlign: "center", marginBottom: "10px" }}>Balloon Trajectories</h3>
        <List>
          {visibleBalloonIds.map(balloon_id => (
            <ListItem key={balloon_id} disablePadding>
              <ListItemButton 
                selected={selectedBalloon === balloon_id} 
                onClick={() => setSelectedBalloon(balloon_id)}
                sx={{ textAlign: "center", color: "#ddd", '&.Mui-selected': { backgroundColor: "#444", color: "#fff" } }}
              >
                <ListItemText primary={`Balloon ${balloon_id}`} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
        <Divider sx={{ background: "#555" }} />
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "10px" }}>
          <Button 
            variant="contained" 
            onClick={() => {setPage(Math.max(page - 1, 0)); setPredictionData()}} 
            disabled={page === 0}
            sx={{ background: "#666", '&:hover': { background: "#555" }}}
          >
            Prev
          </Button>
          <Button 
            variant="contained" 
            onClick={() => {setPage(Math.min(page + 1, totalPages - 1)); setPredictionData() }} 
            disabled={page >= totalPages - 1}
            sx={{ background: "#666", '&:hover': { background: "#555" } }}
          >
            Next
          </Button>
        </div>
        <Button 
          variant="contained" 
          fullWidth 
          onClick={() => {
            setSelectedBalloon(null);
            setShowingPageOnly(true);
          }}
          sx={{ marginTop: "10px", background: "#666", '&:hover': { background: "#555" } }}
        >
          Show All (Current 50)
        </Button>
      </div>
      <div ref={cesiumContainerRef} style={{ flex: 1 }} />
      {selectedPoint && (
        <Dialog open={Boolean(selectedPoint)} onClose={() => setSelectedPoint(null)}>
          <DialogTitle>Point Details</DialogTitle>
          <DialogContent>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Attribute</TableCell>
                    <TableCell>Value</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(selectedPoint).map(([key, value]) => (
                    <TableRow key={key}>
                      <TableCell>{key}</TableCell>
                      <TableCell>{JSON.stringify(value)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </DialogContent>
        </Dialog>
      )}
      
      {selectedBalloon && (
        <Dialog open={Boolean(selectedBalloon)} onClose={() => setSelectedBalloon(null)}>
          <DialogTitle>Trajectory Details</DialogTitle>
          <DialogContent>
          <IconButton
              aria-label="close"
              onClick={() => setSelectedBalloon(null)}
              style={{ position: "absolute", right: 10, top: 10 }}
            >
              <CloseIcon />
            </IconButton>
            <p><strong>Balloon ID:</strong> {selectedBalloon}</p>
            <Button variant="contained" color="primary" onClick={handlePredict}>Predict</Button>
          </DialogContent>
        </Dialog>
      )}
  
    </div>
    
    
  );
}

export default App;
