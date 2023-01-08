import React, { useState, useEffect, useContext } from "react";
import { useParams } from "react-router-dom";
import { UserContext } from "../../App";
export default function UserProfile() {
  const [userProfile, setProfile] = useState([]);
  const { state, dispatch } = useContext(UserContext);
  const { user_id } = useParams();
  const [showfollow, setShowFollow] = useState(
    state ? !state.following.includes(user_id) : true
  );
  //   console.log(user_id);
  useEffect(() => {
    // console.log("hi");
    fetch(`/user/${user_id}`, {
      headers: {
        Authorization: "Bearer " + localStorage.getItem("jwt_NITR"),
      },
    })
      .then((res) => res.json())
      .then((result) => {
        // console.log(result);
        // console.log("hi");
        setProfile(result);
        // console.log(userProfile);
      });
  });
  const followUser = () => {
    fetch("/follow", {
      method: "put",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + localStorage.getItem("jwt_NITR"),
      },
      body: JSON.stringify({
        followId: user_id,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("ji");
        console.log(data);
        dispatch({
          type: "UPDATE",
          payload: { following: data.following, followers: data.followers },
        });
        localStorage.setItem("user", JSON.stringify(data));
        setProfile((prevState) => {
          return {
            ...prevState,
            user: {
              ...prevState.user,
              followers: [...prevState.user.followers, data._id],
            },
          };
        });
        setShowFollow(false);
      });
  };
  const unfollowUser = () => {
    fetch("/unfollow", {
      method: "put",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + localStorage.getItem("jwt_NITR"),
      },
      body: JSON.stringify({
        unfollowId: user_id,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        dispatch({
          type: "UPDATE",
          payload: { following: data.following, followers: data.followers },
        });
        localStorage.setItem("user", JSON.stringify(data));

        setProfile((prevState) => {
          const newFollower = prevState.user.followers.filter(
            (item) => item != data._id
          );
          return {
            ...prevState,
            user: {
              ...prevState.user,
              followers: newFollower,
            },
          };
        });
        setShowFollow(true);
      });
  };
  return (
    <>
      {userProfile.user ? (
        <div style={{ maxWidth: "750px", margin: "0px auto" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-around",
              margin: "18px 0px",
              borderBottom: "1px solid grey",
            }}
          >
            <div>
              <img
                src={userProfile.user.pic}
                alt=""
                style={{
                  width: "160px",
                  height: "160px",
                  borderRadius: "80px",
                }}
              />
            </div>
            <div>
              <h4>{userProfile ? userProfile.user.name : "loading"}</h4>
              <h5>{userProfile.user.email}</h5>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  width: "108%",
                }}
              >
                <h6>{userProfile.posts.length} posts</h6>
                <h6>{userProfile.user.followers.length} followers</h6>
                <h6>{userProfile.user.following.length} following</h6>
                {showfollow ? (
                  <button
                    style={{
                      margin: "10px",
                    }}
                    className="btn waves-effect waves-light #64b5f6 blue darken-1"
                    onClick={() => followUser()}
                  >
                    Follow
                  </button>
                ) : (
                  <button
                    style={{
                      margin: "10px",
                    }}
                    className="btn waves-effect waves-light #64b5f6 blue darken-1"
                    onClick={() => unfollowUser()}
                  >
                    Unfollow
                  </button>
                )}
              </div>
            </div>
          </div>
          <div className="gallery">
            {userProfile.posts.map((item) => {
              return (
                <img key={item._id} className="item" src={item.photo} alt="" />
              );
            })}
          </div>
        </div>
      ) : (
        <h2>loading...!</h2>
      )}
    </>
  );
}
