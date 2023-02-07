package com.example.small_tflite;
//leaf
import static java.lang.Math.round;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.room.processor.Context;

import android.Manifest;
import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private MappedByteBuffer tfliteModel;
    private Button btn_shot,btn_images,btn_test;
    private ImageView igv_show,igv_show2,igv_show3;
    private TextView textView;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probibilityProcessor;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    private static double[] windows;
    private ProgressBar pbar;
    private double[] area = new double[2];
    private boolean flag_coin = false,flag_leaf = false;
    private int [][] paint_arr = new int[324][512];
    private Canvas canvas2;
    Interpreter tflite;
    String TAG = "測試用";
    Uri imgUri;
    Bitmap bmp=null,complete_bmp2=null;
    Bitmap resize_bmp = null;
    float final_anchors[][][] = new float[1][261888][4];
    float [][] meta_value = new float[1][18];
    int h_h = 0;
    int w_w = 0;
    int count_pixel = 0;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_main);
        getSupportActionBar().hide(); //隱藏標題
        btn_shot = (Button) findViewById(R.id.btn_shot);
        btn_images = (Button) findViewById(R.id.btn_images);
        btn_test = (Button)findViewById(R.id.btn_predict);
        igv_show = (ImageView) findViewById(R.id.igv_show);
        textView = (TextView) findViewById(R.id.textView);
        igv_show2 = (ImageView) findViewById(R.id.igv_show2);
        igv_show3 = (ImageView) findViewById(R.id.igv_show3);
        pbar = (ProgressBar) findViewById(R.id.progressBar);
        pbar.setVisibility(View.GONE);
        try {
            tflite = new Interpreter(loadmodelfile(this));
            Log.d(TAG,   "model load success");

        }catch (IOException e){
            Log.d(TAG, "model load fail" + e);
            e.printStackTrace();
        }
        btn_test.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Thread thread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        runOnUiThread(new Runnable() {
                            @Override

                            public void run() {
                                pbar.setVisibility(View.VISIBLE);
                                //igv_show3.setImageBitmap(null);
                                //igv_show.setImageBitmap(null);
                            }
                        });
                    }
                });
                //thread.start();
                Thread thread2 = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                try {
                                    getAnchor(800,1024,0);
                                }catch (Exception e){
                                    Toast.makeText(getApplicationContext(),"未選擇圖片。",Toast.LENGTH_LONG).show();
                                    resetSetting();
                                    pbar.setVisibility(View.GONE);
                                }
                            }
                        });
                    }
                });
                try {
                    thread.start();
                    thread.join();
                    Thread.sleep(100);
                    thread2.start();

                }catch (Exception e){
                    e.printStackTrace();
                }
            }
        });
    }
    private void getAnchor(double min_dim, double max_dim, double min_scale) {
        if (bmp.getWidth() > 324 || bmp.getHeight() > 512){
            bmp = getResizedBitmap(bmp,324,512);

        }
        int[][] backbone_shapes = new int[5][2];
        int stride[] = new int[5];
        double init_scales[] = {32, 64, 128, 256, 512};
        double ratios[] = {0.5, 1, 2};
        int feature_strides[] = {4, 8, 16, 32, 64};
        double scales[] = new double[3];
        double heights[] = new double[3];
        double widths[] = new double[3];
        double anchors[][] = new double[261888][4];
        for (int i = 0; i < 5; i++) {
            stride[i] = 2 * (int) (Math.pow(2, i + 1));
            for (int k = 0; k < 2; k++) {
                backbone_shapes[i][k] = (int) (Math.ceil(1024 / stride[i])); }
            int shape_0 = backbone_shapes[i][0];
            double shift_x[] = new double[shape_0];
            double shift_y[] = new double[shape_0];
            for (int k = 0; k < 3; k++) {
                scales[k] = init_scales[i];
                heights[k] = scales[k] / Math.sqrt(ratios[k]);
                widths[k] = scales[k] * Math.sqrt(ratios[k]); }
            for (int a = 0; a < shape_0; a++) {
                shift_x[a] = a * feature_strides[i]; }
            double box_widths[][] = new double[shape_0 * shape_0][3];
            double box_centers_x[][] = new double[shape_0 * shape_0][3];
            double box_heights[][] = new double[shape_0 * shape_0][3];
            double box_centers_y[][] = new double[shape_0 * shape_0][3];
            for (int x = 0; x < shape_0 * shape_0; x++) {
                for (int y = 0; y < 3; y++) {
                    box_widths[x][y] = widths[y];
                    box_centers_x[x][y] = shift_x[x % shift_x.length];
                    box_heights[x][y] = heights[y];
                    box_centers_y[x][y] = shift_x[x / shift_x.length]; } }
            double flattened_widths[] = new double[shape_0 * shape_0 * 3];
            double flattened_centers_x[] = new double[shape_0 * shape_0 * 3];
            double flattened_heights[] = new double[shape_0 * shape_0 * 3];
            double flattened_centers_y[] = new double[shape_0 * shape_0 * 3];
            int s = 0;
            for (int k = 0; k < shape_0 * shape_0; k++)
                for (int j = 0; j < 3; j++) {
                    flattened_widths[s] = box_widths[k][j];
                    s++; }
            s = 0;
            for (int k = 0; k < shape_0 * shape_0; k++)
                for (int j = 0; j < 3; j++) {
                    flattened_centers_x[s] = box_centers_x[k][j];
                    s++; }
            s = 0;
            for (int k = 0; k < shape_0 * shape_0; k++)
                for (int j = 0; j < 3; j++) {
                    flattened_heights[s] = box_heights[k][j];
                    s++; }
            s = 0;
            for (int k = 0; k < shape_0 * shape_0; k++)
                for (int j = 0; j < 3; j++) {
                    flattened_centers_y[s] = box_centers_y[k][j];
                    s++; }
            double box_sizes[][] = new double[shape_0 * shape_0 * 3][2];
            double box_centers[][] = new double[shape_0 * shape_0 * 3][2];
            for (int m = 0; m < shape_0 * shape_0 * 3; m++) {
                box_sizes[m][0] = flattened_heights[m];
                box_sizes[m][1] = flattened_widths[m];
                box_centers[m][0] = flattened_centers_y[m];
                box_centers[m][1] = flattened_centers_x[m]; }
            double boxes[][] = new double[shape_0 * shape_0 * 3][4];
            for (int n = 0; n < shape_0 * shape_0 * 3; n++) {
                boxes[n][0] = box_centers[n][0] - 0.5 * box_sizes[n][0];
                boxes[n][1] = box_centers[n][1] - 0.5 * box_sizes[n][1];
                boxes[n][2] = box_centers[n][0] + 0.5 * box_sizes[n][0];
                boxes[n][3] = box_centers[n][1] + 0.5 * box_sizes[n][1]; }
            if (i == 0) {
                for (int y = 0; y < shape_0 * shape_0 * 3; y++) {
                    for (int z = 0; z < 4; z++) {
                        anchors[y][z] = boxes[y][z]; } } }
            else if (i == 1) {
                for (int y = 0; y < shape_0 * shape_0 * 3; y++) {
                    for (int z = 0; z < 4; z++) {
                        anchors[y + 196608][z] = boxes[y][z]; } } }
            else if (i == 2) {
                for (int y = 0; y < shape_0 * shape_0 * 3; y++) {
                    for (int z = 0; z < 4; z++) {
                        anchors[y + 196608 + 49152][z] = boxes[y][z]; } } }
            else if (i == 3) {
                for (int y = 0; y < shape_0 * shape_0 * 3; y++) {
                    for (int z = 0; z < 4; z++) {
                        anchors[y + 196608 + 49152 + 12288][z] = boxes[y][z]; } } }
            else { for (int y = 0; y < shape_0 * shape_0 * 3; y++) {
                for (int z = 0; z < 4; z++) {
                    anchors[y + 196608 + 49152 + 12288 + 3072][z] = boxes[y][z]; } } } }
        for (int z = 0; z < 261888; z++) {
            anchors[z][0] = anchors[z][0] - 0;
            anchors[z][1] = anchors[z][1] - 0;
            anchors[z][2] = anchors[z][2] - 1;
            anchors[z][3] = anchors[z][3] - 1; }
        for (int z = 0; z < 261888; z++) {
            for (int c = 0; c < 4; c++) {
                anchors[z][c] = anchors[z][c] / 1023; } }
        //float final_anchors[][][] = new float[1][261888][4];
        for (int z = 0; z < 261888; z++) {
            for (int c = 0; c < 4; c++) {
// final_anchors[0][z][c] = anchors[z][c];
                final_anchors[0][z][c] = (float)anchors[z][c]; } }
        double w = bmp.getWidth();
        double h = bmp.getHeight();
        double height = h;
        double width =w;
        double scale = 1;
        double[] window ={0,0,w,h};
        Log.i("anchor",Arrays.deepToString(final_anchors));
        Log.i("w,h",String.valueOf(w)+" "+String.valueOf(h));
        scale = Math.max(1, min_dim / Math.min(h, w));
        if(scale < min_scale){
            scale = min_scale; }
        double image_max = Math.max(h, w);
        if (round(image_max * scale) > max_dim) {
            Double ab = max_dim;
            double ac = image_max;
            scale = ab / ac; }
        h = h*scale;
        w = w*scale;
        double top_pad = (max_dim - h) / 2;
        double left_pad = (max_dim - w) / 2;
        double right_pad = max_dim - w - left_pad;
        double bottom_pad = max_dim - h - top_pad;
        int[] padding = {(int)top_pad,(int)bottom_pad,(int)left_pad,(int)right_pad};
        resize_bmp = getResizedBitmap(bmp,(int)w,(int)h);
        resize_bmp = imagePadding(resize_bmp,padding);
        window[0] = top_pad; window[1] = left_pad;
        window[2] = h + top_pad;
        window[3] = w + left_pad;
        meta_value[0][0]=0;
        meta_value[0][1] =(float)height;
        meta_value[0][2] =(float)width;
        meta_value[0][3] =3;
        meta_value[0][4] =1024;
        meta_value[0][5] =1024;
        meta_value[0][6] =3;
        meta_value[0][7] =(float)(int)window[0];
        meta_value[0][8] =(float)(int)window[1];
        meta_value[0][9] =(float)(int)window[2];
        meta_value[0][10] =(float)(int)window[3];
        meta_value[0][11] =(float)scale;
        meta_value[0][12] =0;
        meta_value[0][13] = 0;
        meta_value[0][14] =0;
        meta_value[0][15] = 0;
        meta_value[0][16] =0;
        meta_value[0][17] = 0;
        windows = window;
        Log.i("meta_value",Arrays.deepToString(meta_value));
        Log.i("window",Arrays.toString(window));
        h_h = (int)h;
        w_w = (int)w;
        sendvalue(resize_bmp,meta_value, final_anchors,h_h,w_w);
    }

    private Bitmap imagePadding(Bitmap resize_bmp,int[] padding) {
        int[][] bmp_arr = arrayFromBitmap(resize_bmp);
        int[] flatten = new int[bmp_arr.length*bmp_arr[0].length];
        for (int i =0;i<bmp_arr.length;i++){
            for (int j=0;j<bmp_arr[0].length;j++){
                flatten[i + j*bmp_arr.length] = bmp_arr[i][j] ;
            }
        }
        Bitmap padding_bmp = Bitmap.createBitmap(1024,1024, Bitmap.Config.ARGB_8888);
        for(int i=0;i<1024;i++){
            for (int j=0;j<1024;j++){
                if(i < padding[2] | i > bmp_arr.length+padding[3]){
                    padding_bmp.setPixel(i,j,Color.BLACK);
                }else if (j < padding[0] | j > bmp_arr[0].length + padding[1]){
                    padding_bmp.setPixel(i,j,Color.BLACK);
                }
            }
        }
        padding_bmp.setPixels(flatten,0,resize_bmp.getWidth(),padding[2],padding[0],resize_bmp.getWidth(),resize_bmp.getHeight());
        igv_show2.setImageBitmap(padding_bmp);
        return padding_bmp;
    }

    private void sendvalue(Bitmap bitmap,float[][] meta_value, float[][][] final_anchors, int h, int w) {
        int imagetensorindex = 0;
        /*
        DataType imageDatatype = tflite.getInputTensor(imagetensorindex).dataType();
        Log.i(TAG,String.valueOf(imageDatatype));
        inputImageBuffer = new TensorImage(imageDatatype);
        inputImageBuffer = loadImage(bitmap);
         */
        ByteBuffer image = ByteBuffer.allocateDirect(1 * 1024 * 1024 *3*4);
        image.order(ByteOrder.nativeOrder());
        image.rewind();
        int[] intvalue = new int[1024*1024];
        bitmap.getPixels(intvalue,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pix = 0;
        for (int i =0;i<bitmap.getWidth();i++){
            for (int j=0;j<bitmap.getHeight();j++){
                int val =intvalue[pix];
                image.putFloat((((val >> 16) & 0xFF)-123.7f));
                image.putFloat((((val >> 8) & 0xFF)-116.8f));
                image.putFloat((((val) & 0xFF)-103.9f));
                pix++;
            }
        }
        detectObjects2(image,meta_value,final_anchors);//inputImageBuffer.getBuffer()
    }


    public void detectObjects2(ByteBuffer input, float[][] meta_value, float [][][] final_anchors) {
        Object[] inputs = new Object[]{input, meta_value, final_anchors};
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, new float[1][1000][4]);
        outputs.put(1, new float[1][1000][3][4]);
        outputs.put(2, new float[1][1000][3]);
        outputs.put(3, new float[1][100][6]);
        outputs.put(4, new float[1][100][28][28][3]);
        outputs.put(5, new float[1][261888][4]);
        outputs.put(6, new float[1][261888][2]);

        try {
            Date date = Calendar.getInstance().getTime();
            DateFormat formatter = new SimpleDateFormat("dd/MM/yyyy-mm-ss");
            String today = formatter.format(date);
            tflite.runForMultipleInputsOutputs(inputs, outputs);
            float[][][] output3 = (float[][][]) outputs.get(3);
            float[][][][][] output4 = (float[][][][][]) outputs.get(4);
            Log.i("detection",Arrays.deepToString(output3));
            int flag[] = new int[100];
            Date dates = Calendar.getInstance().getTime();
            DateFormat formatterdd = new SimpleDateFormat("dd/MM/yyyy-mm-ss");
            String todays = formatterdd.format(dates);
            int count_object = 0;
            for(int i = 0; i < 100 ; i++){
                if (output3[0][i][4] != 0){
                    count_object++;
                    if (i != 0) {
                        if ((double)Math.round(output3[0][i][0]*100)/100 == (double)Math.round(output3[0][i - 1][0]*100)/100 &
                                (double)Math.round(output3[0][i][1]*100)/100 == (double)Math.round(output3[0][i - 1][1]*100)/100 &
                                (double)Math.round(output3[0][i][2]*100)/100 == (double)Math.round(output3[0][i - 1][2]*100)/100 &
                                (double)Math.round(output3[0][i][3]*100)/100 == (double)Math.round(output3[0][i - 1][3]*100)/100) {
                            flag[i] = -1;
                            //Log.i("mathtest",String.valueOf(Math.round(output3[0][i][0]*10000)));
                            continue;

                        }
                    }
                    flag[i] = (int)output3[0][i][4];
                }
            }
            Log.i("類別",Arrays.toString(flag));
            double boxes[][] = new double[count_object][4];
            int class_ids[] = new int[count_object];
            int scores[] = new int[count_object];
            int masks[][][] = new int[count_object][28][28];
            for (int i = 0; i < count_object; i++){
                if (flag[i] == -1)continue;
                for (int j = 0; j < 6;j++) {
                    if (j < 4) {
                        boxes[i][j] = output3[0][i][j];
                    }
                    if(j == 4){
                        class_ids[i] = Math.round(output3[0][i][j]);
                    }
                    if(j == 5){
                        scores[i] = Math.round(output3[0][i][j]);
                    }

                }
            }


            for (int i = 0; i < flag.length;i++){
                if (flag[i] == 1){
                    flag_coin = true;
                }
                if (flag[i] == 2){
                    flag_leaf = true;
                }
            }

            Log.i("flag_coin",String.valueOf(flag_coin));
            Log.i("flag_leaf",String.valueOf(flag_leaf));
            if (flag_coin == false && flag_leaf == false){
                Toast.makeText(getApplicationContext(),"未找出葉片或硬幣。",Toast.LENGTH_LONG).show();
                resetSetting();
                pbar.setVisibility(View.GONE);
            }else {
                if (flag_coin == false || flag_leaf == false) {
                    if (area[0] == 0) {
                        for (int i = 0; i < count_object; i++) {
                            if (flag[i] == -1) continue;
                            for (int j = 0; j < 28; j++) {
                                for (int k = 0; k < 28; k++) {
                                    if (Math.round(output4[0][i][j][k][class_ids[i]]) > 0.5) {
                                        //masks[i][j][k] = 1;
                                        masks[i][k][j] = Color.WHITE;
                                    } else {
                                        //masks[i][j][k] = 0;
                                        masks[i][k][j] = Color.BLACK;
                                    }

                                }
                            }
                        }

                        double[] norm = {(windows[0] - 0) / 1023, (windows[1] - 0) / 1023, (windows[2] - 1) / 1023, (windows[3] - 1) / 1023};
                        Log.i("norm", Arrays.toString(norm));
                        double wy1 = norm[0];
                        double wx1 = norm[1];
                        double wy2 = norm[2];
                        double wx2 = norm[3];
                        double wh = wy2 - wy1;
                        double ww = wx2 - wx1;
                        double[] shift = {wy1, wx1, wy1, wx1};
                        double[] shift2 = {0, 0, 1, 1};
                        double[] scale = {meta_value[0][1] - 1, meta_value[0][2] - 1, meta_value[0][1] - 1, meta_value[0][2] - 1};
                        double[] scale2 = {wh, ww, wh, ww};
                        int[] originshape = {(int) meta_value[0][2], (int) meta_value[0][1]};
                        Log.i("boxesoutput", Arrays.deepToString(boxes));
                        for (int b_i = 0; b_i < count_object; b_i++) {
                            if (flag[b_i] == -1) continue;
                            for (int b_j = 0; b_j < 4; b_j++) {
                                boxes[b_i][b_j] = Math.round(((boxes[b_i][b_j] - shift[b_j]) / scale2[b_j]) * scale[b_j] + shift2[b_j]);
                            }
                        }
                        Log.i("boxes", Arrays.deepToString(boxes));
                        int full_masks1[][];
                        int full_masks2[][][] = new int[count_object][originshape[0]][originshape[1]];
                        for (int i = 0; i < count_object; i++) {
                            if (flag[i] == -1) continue;
                            full_masks1 = unmold_mask(masks[i], boxes[i], originshape, count_object);
                            for (int f_j = 0; f_j < originshape[0]; f_j++) {
                                for (int f_k = 0; f_k < originshape[1]; f_k++) {
                                    full_masks2[i][f_j][f_k] = full_masks1[f_j][f_k];
                                }
                            }
                        }
                        area = countarea(full_masks2, class_ids);
                        visuallize(full_masks2, class_ids, scores, originshape, boxes);
                        BitmapDrawable drawable = (BitmapDrawable) igv_show3.getDrawable();
                        Bitmap bitmap_show3 = drawable.getBitmap();
                        bmp = bitmap_show3;
                        Handler handler = new Handler();
                        handler.postDelayed(new Runnable() {
                            public void run() {
                                // Actions to do after 10 seconds
                                getAnchor(800, 1024, 0);
                                resetSetting();
                                pbar.setVisibility(View.GONE);
                            }
                        }, 1000);
                        //getAnchor(800,1024,0);
                        Bitmap screenshot = getScreenShot();
                        saveBitmap(screenshot);
                        Log.i(TAG, "結束");
                    } else {
                        Toast.makeText(getApplicationContext(), "未找出葉片或硬幣。", Toast.LENGTH_LONG).show();
                        resetSetting();
                        pbar.setVisibility(View.GONE);
                    }
                } else {
                    for (int i = 0; i < count_object; i++) {
                        if (flag[i] == -1) continue;
                        for (int j = 0; j < 28; j++) {
                            for (int k = 0; k < 28; k++) {
                                if (Math.round(output4[0][i][j][k][class_ids[i]]) > 0.5) {
                                    //masks[i][j][k] = 1;
                                    masks[i][k][j] = Color.WHITE;
                                } else {
                                    //masks[i][j][k] = 0;
                                    masks[i][k][j] = Color.BLACK;
                                }

                            }
                        }
                    }
                    double[] norm = {(windows[0] - 0) / 1023, (windows[1] - 0) / 1023, (windows[2] - 1) / 1023, (windows[3] - 1) / 1023};
                    Log.i("norm", Arrays.toString(norm));
                    double wy1 = norm[0];
                    double wx1 = norm[1];
                    double wy2 = norm[2];
                    double wx2 = norm[3];
                    double wh = wy2 - wy1;
                    double ww = wx2 - wx1;
                    double[] shift = {wy1, wx1, wy1, wx1};
                    double[] shift2 = {0, 0, 1, 1};
                    double[] scale = {meta_value[0][1] - 1, meta_value[0][2] - 1, meta_value[0][1] - 1, meta_value[0][2] - 1};
                    double[] scale2 = {wh, ww, wh, ww};
                    int[] originshape = {(int) meta_value[0][2], (int) meta_value[0][1]};
                    Log.i("boxesoutput", Arrays.deepToString(boxes));
                    for (int b_i = 0; b_i < count_object; b_i++) {
                        if (flag[b_i] == -1) continue;
                        for (int b_j = 0; b_j < 4; b_j++) {
                            boxes[b_i][b_j] = Math.round(((boxes[b_i][b_j] - shift[b_j]) / scale2[b_j]) * scale[b_j] + shift2[b_j]);
                        }
                    }
                    Log.i("boxes", Arrays.deepToString(boxes));
                    int full_masks1[][];
                    int full_masks2[][][] = new int[count_object][originshape[0]][originshape[1]];
                    for (int i = 0; i < count_object; i++) {
                        if (flag[i] == -1) continue;
                        full_masks1 = unmold_mask(masks[i], boxes[i], originshape, count_object);
                        for (int f_j = 0; f_j < originshape[0]; f_j++) {
                            for (int f_k = 0; f_k < originshape[1]; f_k++) {
                                full_masks2[i][f_j][f_k] = full_masks1[f_j][f_k];
                            }
                        }
                    }
                    visuallize(full_masks2, class_ids, scores, originshape, boxes);
                    countarea(full_masks2, class_ids);
                    Bitmap screenshot = getScreenShot();
                    saveBitmap(screenshot);
                    resetSetting();
                    pbar.setVisibility(View.GONE);
                    Log.i(TAG, "結束2");
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(getApplicationContext(),"未找出葉片或硬幣。",Toast.LENGTH_LONG).show();
            resetSetting();
            pbar.setVisibility(View.GONE);
        }
    }

    private double[] countarea(int[][][] masks,int[] class_ids) {
        int leaf_area = 0;
        int coin_area = 0;
        for (int i = 0; i < masks.length; i++) {
            for (int j = 0; j < masks[0].length; j++) {
                for (int k = 0; k < masks[0][0].length; k++) {
                    if (masks[i][j][k] != Color.TRANSPARENT){
                        if (class_ids[i] != 1){
                            leaf_area++;
                        }else {
                            coin_area++;
                        }
                    }
                }
            }
        }
        Log.i("area_test",String.valueOf(leaf_area));
        Log.i("area_test",String.valueOf(coin_area));
        double[] tmp_area = {0,0};
        if (flag_leaf == true && flag_coin == true && area[0] == 0){
            tmp_area[0] = (double)leaf_area/(double)coin_area*3.14;
            tmp_area[1] = 2; //0,1,2
            textView.setText(String.format("%.2f",tmp_area[0]));
            Log.i("area_test","first");
        }else{
            if (area[0] == 0) {
                if (flag_coin == true) {
                    tmp_area[0] = coin_area;
                    tmp_area[1] = 0;
                    Log.i("area_test","first_coin");
                }
                if (flag_leaf == true) {
                    tmp_area[0] = leaf_area;
                    tmp_area[1] = 1;
                    Log.i("area_test","first_leaf");
                }
            }else{
                if (area[1] == 0){
                    tmp_area[0] = (double)leaf_area / area[0];
                    Log.i("area_test","second_leaf");
                }else{
                    tmp_area[0] = area[0] / (double)coin_area;
                    Log.i("area_test","second_coin");
                }
                textView.setText(String.format("%.2f",tmp_area[0]));
            }
        }
        Log.i("area_test",Arrays.toString(tmp_area));
        return tmp_area;
    }

    private void visuallize(int[][][] masks, int[] class_ids, int[] scores,int[] origin,double[][] boxes) {
        Bitmap origin_bmp = bmp;
        Bitmap test_bmp = Bitmap.createBitmap(origin[0],origin[1], Bitmap.Config.ARGB_8888);
        String[] label = {"BG", "coin","leaf"};
        int[][] origin_arr = arrayFromBitmap(origin_bmp);
        for (int i = 0; i < masks.length; i++) {
            for (int j = 0; j < origin[0]; j++) {
                for (int k = 0; k < origin[1]; k++) {
                    if (masks[i][j][k] != Color.TRANSPARENT){
                        origin_arr[j][k] = masks[i][j][k];
                        paint_arr[j][k] = masks[i][j][k];
                    }
                }
            }
        }

        if (flag_coin == false || flag_leaf == false){
            complete_bmp2 = bitmapFromArray(paint_arr).copy(Bitmap.Config.ARGB_8888,true);
        }
        if (flag_coin == true || flag_leaf == true){
            complete_bmp2 = bitmapFromArray(origin_arr).copy(Bitmap.Config.ARGB_8888,true);
        }
       //complete_bmp2 = bitmapFromArray(paint_arr).copy(Bitmap.Config.ARGB_8888,true);
        Canvas canvas = new Canvas(complete_bmp2);

        Paint paint = new Paint();
        for (int i = 0; i < boxes.length; i++) {
            int left = (int) boxes[i][1];
            int top = (int) boxes[i][0];
            int right = (int) boxes[i][3];
            int bottom = (int) boxes[i][2];
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);//不填充
            paint.setStrokeWidth(1);  //线的宽度
            canvas.drawRect(left, top, right, bottom, paint);
            paint.setTextSize(15); //設置文字大小
            // 抗锯齿
            paint.setAntiAlias(true);
            // 防抖动
            paint.setDither(true);
            paint.setColor(Color.parseColor("#ff0000"));
            canvas.drawText(String.valueOf(label[class_ids[i]]), (left), (top - 8), paint);
        }
        Bitmap complete_bmp = bitmapFromArray(origin_arr).copy(Bitmap.Config.ARGB_8888,true);
        igv_show3.setImageBitmap(complete_bmp);
        //igv_show2.setImageBitmap(origin_bmp);
        igv_show.setImageBitmap(complete_bmp2);


    }
    private int[][] unmold_mask(int[][] mask,double[] box,int[] origin,int count_object){
        int y1 = (int)box[0];
        int x1 = (int)box[1];
        int y2 = (int)box[2];
        int x2 = (int)box[3];
        Bitmap mask_bmp = bitmapFromArray(mask);
        mask_bmp = getResizedBitmap(mask_bmp,Math.abs((int)(x2-x1)),Math.abs((int)(y2-y1)));
        //Log.i("testmask",Arrays.deepToString(obejct_mask));
        int full_mask[][] = new int[origin[0]][origin[1]];
        int[] color = new int[(int)(y2 - y1)*(int)(x2 - x1)];
        int[][] obeject_mask3 = new int[(int)(x2 - x1)][(int)(y2 - y1)];
        mask_bmp.getPixels(color,0,Math.abs((int)(x2 - x1)),0,0,Math.abs((int)(x2 - x1)),Math.abs((int)(y2 - y1)));
        for (int i =0;i<mask_bmp.getWidth();i++){
            for (int j=0;j<mask_bmp.getHeight();j++){
                obeject_mask3[i][j] = color[i + j*mask_bmp.getWidth()];
            }
        }
        Bitmap empty_bmp = Bitmap.createBitmap(origin[0],origin[1], Bitmap.Config.ARGB_8888);
        int[][] test_arr = arrayFromBitmap(empty_bmp );
        for(int i = 0; i < origin[0];i++){
            for (int j = 0; j < origin[1]; j++){
                if(i >= (int)x1 & i < (int)x2 & j >= (int)y1 & j<(int)y2){
                    if(color[i-x1+(j-y1)*((int)(x2-x1))] == Color.WHITE) {
                        //test_arr[i][j] = color[i - x1 + (j - y1) * ((int) (x2 - x1))];
                        if (flag_coin == false){
                            test_arr[i][j] = Color.parseColor("#D2AE2755");
                        }else{
                            test_arr[i][j] = Color.parseColor("#D2AE2755");
                        }
                    }else {
                        test_arr[i][j] = 0;
                    }
                }
            }
        }
        return test_arr;
    }

    public static Bitmap bitmapFromArray(int[][] pixels2d){
        int width = pixels2d.length;
        int height = pixels2d[0].length;
        int[] pixels = new int[width * height];
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                pixels[i + j*width] = pixels2d[i][j];
            }
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
    }

    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // CREATE A MATRIX FOR THE MANIPULATION
        Matrix matrix = new Matrix();
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight);

        // "RECREATE" THE NEW BITMAP
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
     //   bm.recycle();
        return resizedBitmap;
    }

    public static int[][] arrayFromBitmap(Bitmap source){
        int width = source.getWidth();
        int height = source.getHeight();
        int[][] result = new int[width][height];
        int[] pixels = new int[width*height];
        source.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                result[i][j] =  pixels[i + j*width];
            }
        }
        //Log.i("array",Arrays.deepToString(result));
        return result;
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);
        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        //.add(new NormalizeOp((float)127.5,(float)127.5,(float)127.5))
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }
    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor=activity.getAssets().openFd("modelleaf0225.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    public void onGet(View v) throws IOException {
        if (ActivityCompat.checkSelfPermission(this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    200);
        }
        else {
            savePhoto();
        }
    }

    protected void onActivityResult (int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(resultCode == Activity.RESULT_OK) {   //要求的意圖成功了
            switch(requestCode) {
                case 100: //拍照
                    Intent it = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, imgUri);//設為系統共享媒體檔
                    sendBroadcast(it);
                    break;
                case 101: //選取相片
                    imgUri = data.getData();  //取得選取相片的 Uri
                    break;
            }
            showImg();  //顯示 imgUri 所指明的相片
        }
        else {
            Toast.makeText(this, requestCode==100? "沒有拍到照片":
                    "沒有選取相片", Toast.LENGTH_LONG)
                    .show();
        }
    }

    private void savePhoto () {
        imgUri =  getContentResolver().insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                new ContentValues());
        Intent it = new Intent("android.media.action.IMAGE_CAPTURE");
        it.putExtra(MediaStore.EXTRA_OUTPUT, imgUri);    //將 uri 加到拍照 Intent 的額外資料中
        startActivityForResult(it, 100);
    }
    public void onPick(View v) {
        Intent it = new Intent(Intent.ACTION_GET_CONTENT);    //動作設為 "選取內容"
        it.setType("image/*");            //設定要選取的媒體類型為：所有類型的圖片
        startActivityForResult(it, 101);  //啟動意圖, 並要求傳回選取的圖檔
    }
    void showImg() {
        int iw, ih, vw, vh;
        boolean needRotate;  //用來儲存是否需要旋轉

        BitmapFactory.Options option = new BitmapFactory.Options(); //建立選項物件
        option.inJustDecodeBounds = true;      //設定選項：只讀取圖檔資訊而不載入圖檔

        //讀取圖檔資訊存入 Option 中
        try {
            BitmapFactory.decodeStream(getContentResolver().openInputStream(imgUri), null, option);
        }
        catch (IOException e) {
            Toast.makeText(this, "讀取照片資訊時發生錯誤", Toast.LENGTH_LONG).show();
            return;
        }

        iw = option.outWidth;   //由 option 中讀出圖檔寬度
        ih = option.outHeight;  //由 option 中讀出圖檔高度
        vw = igv_show.getWidth();    //取得 ImageView 的寬度
        vh = igv_show.getHeight();   //取得 ImageView 的高度

        int scaleFactor;
        if(iw<ih) {    //如果圖片的寬度小於高度
            needRotate = false;       				//不需要旋轉
            scaleFactor = Math.min(iw/vw, ih/vh);   // 計算縮小比率
        }
        else {
            needRotate = true;       				//需要旋轉
            scaleFactor = Math.min(iw/vh, ih/vw);   // 將 ImageView 的寬、高互換來計算縮小比率
        }

        option.inJustDecodeBounds = false;  //關閉只載入圖檔資訊的選項
        option.inSampleSize = scaleFactor;  //設定縮小比例, 例如 2 則長寬都將縮小為原來的 1/2

        //載入圖檔

        try {
            bmp = BitmapFactory.decodeStream(getContentResolver().openInputStream(imgUri), null, option);
        } catch (IOException e) {
            Toast.makeText(this, "無法取得照片", Toast.LENGTH_LONG).show();
        }

        if(needRotate) { //如果需要旋轉
            Matrix matrix = new Matrix();  //建立 Matrix 物件
            matrix.postRotate(90);         //設定旋轉角度
            bmp = Bitmap.createBitmap(bmp , //用原來的 Bitmap 產生一個新的 Bitmap
                    0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
        }

        igv_show.setImageBitmap(bmp); //顯示圖片
        igv_show2.setImageBitmap(bmp);
        igv_show3.setImageBitmap(null);
        textView.setText("0");
    }
    private void resetSetting(){
        flag_coin = false;
        flag_leaf = false;
        paint_arr = new int[324][512];
        area = new double[2];
    }
    private Bitmap getScreenShot(){
        //將螢幕畫面存成一個View
        View view = getWindow().getDecorView();
        view.setDrawingCacheEnabled(true);
        view.buildDrawingCache();
        Bitmap fullBitmap = view.getDrawingCache();
        //取得系統狀態欄高度
        Rect rect = new Rect();
        getWindow().getDecorView().getWindowVisibleDisplayFrame(rect);
        int statusBarHeight = rect.top;
        //取得手機長、寬
        int phoneWidth = getWindowManager().getDefaultDisplay().getWidth();
        int phoneHeight = getWindowManager().getDefaultDisplay().getHeight();
        //將螢幕快取到的圖片修剪尺寸(去掉status bar)後，存成Bitmap
        Bitmap bitmap = Bitmap.createBitmap(fullBitmap,0,statusBarHeight,phoneWidth
                ,phoneHeight-statusBarHeight);
        //清除螢幕截圖快取，避免內存洩漏
        view.destroyDrawingCache();
        return bitmap;

    }

    private void saveBitmap(Bitmap bmp) throws IOException{
        boolean saved;
        OutputStream fos;
        Date currentTime = Calendar.getInstance().getTime();
        SimpleDateFormat date = new SimpleDateFormat("yyyyMMddHHmmss");
        String name = date.format(currentTime)+"_leaf";
        String IMAGES_FOLDER_NAME = "leave";
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ContentResolver resolver = this.getContentResolver();
            ContentValues contentValues = new ContentValues();
            contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
            contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "image/png");
            contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/" + IMAGES_FOLDER_NAME);
            Uri imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
            fos = resolver.openOutputStream(imageUri);
        } else {
            String imagesDir = Environment.getExternalStoragePublicDirectory(
                    Environment.DIRECTORY_DCIM).toString() + File.separator + IMAGES_FOLDER_NAME;

            File file = new File(imagesDir);

            if (!file.exists()) {
                file.mkdir();
            }

            File image = new File(imagesDir, name + ".png");
            fos = new FileOutputStream(image);

        }

        saved = bmp.compress(Bitmap.CompressFormat.PNG, 100, fos);
        fos.flush();
        fos.close();
    }

}