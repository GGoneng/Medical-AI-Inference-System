import { useRef, useState } from "react";
import axios from "axios";

const Chat = ({ file, setFile, setLoading, id, setID, url}) => {
    const textareaRef = useRef(null);
    const wrapperRef = useRef(null);

    const [text, setText] = useState("");

    const isDisable = text.trim().length === 0 && !file;

    const handleText = (text) => {
        setText(text);
    };

    const handleTextselect = (e) => {
        handleText(e.target.value);
    };

    const handleScroll = () => {
        const textarea = textareaRef.current;
        const wrapper = wrapperRef.current;
        if (textarea && wrapper) {
            textarea.style.height = "auto";

            const scroll = textarea.scrollHeight;
            const maxHeight = 300;

            if (scroll > maxHeight) {
                textarea.style.height = maxHeight + "px";
                textarea.style.overflowY = "scroll";
            } else {
                textarea.style.height = scroll + "px";
                textarea.style.overflowY = "hidden";
            }

            wrapper.style.height = parseInt(textarea.style.height) + 12 + "px";
        }
    };

    const uploadBackend = async (id, file, text, url) => {
        const form = new FormData();
        

        if (file) form.append("file", file);

        if (text) form.append("text", text);

        if (id) form.append("id", id);

        setLoading(true);

        await axios.post(`${url}/upload`, form, {
            headers: {"Content-Type": "multipart/form-data"}
        })
        .then(response => {
            console.log("서버 응답 : ", response.data);

            if (response.data.id) setID(response.data.id);

            setText("");
            setFile(null);
            setTimeout(() => handleScroll(), 0);
            setLoading(false);
        }
    )
        .catch(err => {
            console.error("업로드 실패 : ", err);
            setLoading(false);
        }
    )};


    return (
        <div ref={wrapperRef} className="flex fixed items-center rounded-[28px] h-auto w-[600px] bottom-[0px] mb-[30px] shadow-chat left-1/2 -translate-x-1/2 bg-white">
            <textarea ref={textareaRef} onInput={handleScroll} value={text} onChange={handleTextselect} className="h-[40px] w-[87%] border-none outline-none text-[15px] box-border pt-2 pl-1 mt-[1.7%] ml-5 bg-transparent leading-none overflow-hidden resize-none"
                                             placeholder="궁금하신 증상을 말씀해주세요"></textarea>
            <button disabled={isDisable} onClick={() => uploadBackend(id, file, text, url)} className={`fixed right-2 bottom-1.5 flex items-center justify-center h-9 w-9 rounded-full text-white 
                ${isDisable ? 'bg-gray-200 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}`}>
                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor" xmlns="http://www.w3.org/2000/svg" className="icon">   
                    <path d="M8.99992 16V6.41407L5.70696 9.70704C5.31643 10.0976 4.68342 10.0976 4.29289 9.70704C3.90237 9.31652 3.90237 8.6835 4.29289 8.29298L9.29289 3.29298L9.36907 3.22462C9.76184 2.90427 10.3408 2.92686 10.707 3.29298L15.707 8.29298L15.7753 8.36915C16.0957 8.76192 16.0731 9.34092 15.707 9.70704C15.3408 10.0732 14.7618 10.0958 14.3691 9.7754L14.2929 9.70704L10.9999 6.41407V16C10.9999 16.5523 10.5522 17 9.99992 17C9.44764 17 8.99992 16.5523 8.99992 16Z"></path>
                </svg>
            </button>
        </div>
    )
};

export default Chat;