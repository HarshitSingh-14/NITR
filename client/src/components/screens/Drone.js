import React, { useState, useEffect, useContext } from "react";
import Select from "react-select";
import { Link } from "react-router-dom";
import { UserContext } from "../../App";
import background from "./back.jpg";
import M from "materialize-css";

export default function Drone() {
  const [role, setRole] = useState(null);
  const [drone, setDrone] = useState([]);
  const { state, dispatch } = useContext(UserContext);
  const options = [
    { value: "1", label: "Distance" },
    { value: "0", label: "Percentage" },
  ];
  useEffect(() => {
    if (role != null) {
      postDetails();
    }
  }, [role]);
  let goTO = (lat, lon) => {
    window.open(
      `https://www.google.com/maps/dir//${lat},${lon}/@${lat},${lon},19z`
    );
    //https://www.google.com/maps/dir/${lat},${lon}//@${lat},${lon},19z
  };
  const postDetails = () => {
    fetch("/getDrone", {
      method: "post",
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => res.json())
      .then(async (data) => {
        if (data.me) {
          console.log(data.me);
        }
      });
    getPosts();
  };
  const getPosts = () => {
    fetch("/getDronePosts", {
      method: "post",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        temp: role,
      }),
    })
      .then((res) => res.json())
      .then(async (data) => {
        //   if (data.me) {
        console.log(data.me);
        setDrone(data.me);
        //   }
      });
  };
  const checkIn = (id, lat, lon, per) => {
    fetch("/deleteDrone", {
      method: "post",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        temp: id,
      }),
    })
      .then((res) => res.json())
      .then(async (data) => {
        if (data.me) {
          console.log(data.me);
        }
        fetch("/sendSms", {
          method: "post",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            name: state.name,
            num: state.num,
            role: state.role,
            loc: `https://www.google.com/maps/dir//${lat},${lon}/@${lat},${lon},19z`,
            percentage: per,
          }),
        })
          .then((res) => res.json())
          .then(async (data) => {
            if (data.me) {
              console.log(data.me);
            }
          });
      });
    // console.log(id);
    M.toast({
      html: "Hotspot Cleaned",
      classes: "toast-container green darken-1",
    });
  };

  return (
    <div>
      <div
        style={{
          width: "300px",
          height: "10px",
          float: "right",
          marginTop: "8px",
        }}
      >
        <Select
          defaultValue={role}
          maxControlHeight={190}
          onChange={(e) => setRole(e.value)}
          options={options}
          theme={(theme) => ({
            ...theme,
            borderRadius: 0,
            colors: {
              ...theme.colors,
              primary25: "#b5fe83",
              primary: "grey",
            },
          })}
        />
      </div>
      <div style={{ paddingTop: "27px" }}>
        {drone.map((item) => {
          return (
            <div
              className="card home-card"
              key={item.id}
              style={{
                padding: "2px",
              }}
            >
              <h5 style={{ paddingLeft: "20px" }}>
                <Link
                  to={
                    "////"
                    //   item.postedBy._id !== state._id
                    //     ? "/profile/" + item.postedBy._id
                    //     : "/profile"
                  }
                >
                  {/* {item.postedBy.name} */}Drone@471
                </Link>{" "}
                {/* {item.postedBy._id == state._id && (
            <i
              className="material-icons"
              style={{
                float: "right",
              }}
              onClick={() => deletePost(item._id)}
            >
              delete
            </i>
          )} */}
              </h5>
              <div className="card-image" style={{ margin: "2px" }}>
                <img src={item.image} alt="" />
              </div>
              <div className="card-content">
                <h6 style={{ fontSize: "30px" }}>
                  <i
                    className="material-icons Large"
                    style={{ fontSize: "50px" }}
                    onClick={() => goTO(item.latitude, item.longitude)}
                  >
                    location_on
                  </i>{" "}
                  <span style={{ fontSize: "30px" }}>{item.distance}m</span>{" "}
                  <span style={{ fontSize: "40px", float: "right" }}>
                    {" "}
                    {item.percentage}%
                  </span>
                </h6>
                <button
                  type="button"
                  id="nav-btn2"
                  className="btn"
                  onClick={() => {
                    checkIn(
                      item.id,
                      item.latitude,
                      item.longitude,
                      item.percentage
                    );
                  }}
                >
                  checkIn
                </button>
                {/* {item.comments.map((record) => {
            return (
              <h6 key={record._id}>
                <span style={{ fontWeight: "500" }}>
                  {record.postedBy.name}
                </span>{" "}
                {record.text}
              </h6>
            );
          })} */}
                {/* <form
            onSubmit={(e) => {
              e.preventDefault();
              //   makeComment(e.target[0].value, item._id);
            }}
          >
            <input type="text" placeholder="add an comment" />
          </form> */}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
