FROM public.ecr.aws/lambda/python:3.9

RUN  pip3 install transformers["ja"] numpy sentencepiece --target "${LAMBDA_TASK_ROOT}" && \
     pip3 install torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html --target "${LAMBDA_TASK_ROOT}"

COPY model ${LAMBDA_TASK_ROOT}/model
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
CMD [ "lambda_function.inference" ] 
