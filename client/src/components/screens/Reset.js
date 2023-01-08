import React, { useState, useContext } from "react";
import { UserContext } from "./../../App";
import { Link, useNavigate } from "react-router-dom";
import M from "materialize-css";
export default function Reset() {
  // console.log(UserContext);
  const Navigate = useNavigate();
  const [email, setEmail] = useState("");
  //Cors or cross origin resource sharing has to be used for backend and frontend as frontend is on 3000 and backend is at 5000 ...
  //well now we have something better
  const PostData = () => {
    if (
      /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/.test(
        email
      )
    ) {
      fetch("/reset-password", {
        method: "post",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          //   console.log(data);
          if (data.error) {
            M.toast({
              html: data.error,
              classes: "toast-container red darken-3",
            });
          } else {
            M.toast({
              html: data.message,
              classes: "toast-container green darken-1",
            });
            Navigate("/signin");
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
    <div className="mycard">
      <div className="card auth-card input-field">
        <h2>Instagram</h2>
        <input
          type="text"
          placeholder="email"
          value={email}
          onChange={(e) => {
            setEmail(e.target.value);
          }}
        />
        <button
          className="btn waves-effect waves-light  blue  darken-1"
          onClick={() => PostData()}
        >
          reset password
        </button>
      </div>
    </div>
  );
}
