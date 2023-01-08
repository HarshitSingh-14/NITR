import React, { useState, useContext } from "react";
import { UserContext } from "./../../App";
import { Link, useNavigate } from "react-router-dom";
import M from "materialize-css";
import videoBg from "./vid.mp4";
import "../../App.css";
export default function Signin() {
  // console.log(UserContext);
  const { state, dispatch } = useContext(UserContext);
  const Navigate = useNavigate();
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  //Cors or cross origin resource sharing has to be used for backend and frontend as frontend is on 3000 and backend is at 5000 ...
  //well now we have something better
  const PostData = () => {
    if (
      /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/.test(
        email
      )
    ) {
      fetch("/signin", {
        method: "post",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          password,
          email,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log(data);
          if (data.error) {
            M.toast({
              html: data.error,
              classes: "toast-container red darken-3",
            });
          } else {
            localStorage.setItem("jwt_NITR", data.token);
            localStorage.setItem("user_NITR", JSON.stringify(data.user));
            dispatch({ type: "USER", payload: data.user });
            M.toast({
              html: "Signed In",
              classes: "toast-container green darken-1",
            });
            Navigate("/");
          }
        })
        .catch((err) => {
          console.log(err);
        });
    } else {
      M.toast({
        html: "Invalid email",
        classes: "toast-container red darken-3",
      });
    }
  };
  return (
    <>
      {/* <video
        src="https://tenor.com/R8C4.gif"
        autoplay
        loop
        playsinline
        muted
        style={{
          objectFit: "cover",
          width: "100vw",
          height: "100vh",
          position: "fixed",
          top: "0",
          left: "0",
        }}
      ></video> */}
      <video
        src={videoBg}
        autoPlay
        loop
        muted
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
        }}
      />
      <div className="mycard content ">
        <div className="card auth-card input-field " style={{ opacity: "0.9" }}>
          <h2 style={{ color: "black" }}>Dr. cleaner</h2>
          <input
            type="text"
            placeholder="email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
            }}
          />
          <input
            type="password"
            placeholder="password"
            value={password}
            onChange={(e) => {
              setPassword(e.target.value);
            }}
          />
          <button
            className="btn waves-effect waves-light  blue  darken-1"
            onClick={() => PostData()}
          >
            Login
          </button>
          {/* <h6>
            <Link to="/signup">Don't have an account ?</Link>
          </h6> */}
          <h6>
            <Link to="/reset">Forgot Password ?</Link>
          </h6>
        </div>
      </div>
    </>
  );
}
