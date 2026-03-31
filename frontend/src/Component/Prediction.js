import axios from "axios";
import { useRef, useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";

const Prediction = ({ id, setLoading, url }) => {
    const [outputs, setOutputs] = useState({ vision: [], llm: [] });
    const [animatedText, setAnimatedText] = useState(null);
    const lastLLMRef = useRef(null);
    const pollingRef = useRef(true);

    const animateText = async (fullText) => {
        setAnimatedText("");
        for (let i = 0; i < fullText.length; i++) {
            setAnimatedText((prev) => prev + fullText[i]);
            await new Promise((resolve) => setTimeout(resolve, 20));
        }
    };

    useEffect(() => {
        if (!id) return;

        pollingRef.current = true;
        setLoading(true);

        const fetchPrediction = async () => {
            if (!pollingRef.current) return;

            try {
                const [visionRes, llmRes] = await Promise.all([
                    axios.get(`${url}/visionOutputs/${id}`),
                    axios.get(`${url}/llmOutputs/${id}`)
                ]);

                const visionOut = visionRes.data.outputs || [];
                const llmOut = llmRes.data.outputs || [];

                if (visionOut.length === 0 && llmOut.length === 0) return;

                setOutputs({ vision: visionOut, llm: llmOut });
                setLoading(false);

                if (llmOut.length > 0) {
                    const newText = llmOut.join("\n\n");

                    if (newText !== lastLLMRef.current) {
                        lastLLMRef.current = newText;
                        animateText(newText);
                    }
                }

            } catch (err) {
                console.error("예측 실패 : ", err);
                setLoading(false);
                pollingRef.current = false;
            }
        };

        const interval = setInterval(fetchPrediction, 2000);

        return () => {
            pollingRef.current = false;
            clearInterval(interval);
        };
    }, [id, setLoading, url]);

    if (!outputs.vision.length && !outputs.llm.length && !animatedText) return null;

    return (
        <div className="relative flex flex-col justify-center items-center mt-20 rounded-[28px] shadow-preview w-[550px] min-h-[550px] p-4">
            {outputs.vision.length > 0 && (
                <div className="flex justify-center items-center mb-5">
                    {outputs.vision.map((base64Img, idx) => (
                        <img
                            key={idx}
                            src={`data:image/png;base64,${base64Img}`}
                            alt={`Vision Prediction ${idx}`}
                            className="max-w-[500px] max-h-[500px] rounded-lg border"
                        />
                    ))}
                </div>
            )}

            {(animatedText || outputs.llm.length > 0) && (
                <div className="p-5 text-left whitespace-pre-wrap w-full border-t border-gray-300 overflow-y-auto">
                    <ReactMarkdown>{animatedText ?? ""}</ReactMarkdown>
                </div>
            )}
        </div>
    );
};

export default Prediction;