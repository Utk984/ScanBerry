package com.example.scanberry;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.scanberry.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result, con_text, con_per, cap, check, time, info;
    ImageView imageView, capture;
    Button picture;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        con_text = findViewById(R.id.con_text);
        con_per = findViewById(R.id.con_per);
        imageView = findViewById(R.id.imageView);
        check = findViewById(R.id.check);
        time = findViewById(R.id.time);
        info = findViewById(R.id.Info);
    }
    public void work(View view) {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(cameraIntent, 1);
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
        }
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocate(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0,image.getWidth(),0,0,image.getWidth(),image.getHeight());
            int pixel = 0;
            for(int i=0;i<imageSize;i++){
                for(int j=0;j<imageSize;j++){
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxpos = 0; float maxcon = 0;
            for(int i=0;i<confidences.length;i++){
                if(confidences[i]>maxcon) {
                    maxcon = confidences[i];
                    maxpos = i;
                }
            }

            String[] classes = {"Under-Ripe", "Ripe", "Very-ripe"};
            String[] colours = {"#F38E8E", "#A4E1A4", "#000000"};
            String[] left = {"2 weeks","2-7 days","2-3 days"};
            String[] inf = {"Most Starch resistant\nPrebiotics\nHigh Fiber",
                    "Sweeter\nHigh level antioxidants\nDigests Quicker",
                    "Sweetest\nHighest level antioxidants\nBest for baking"};
            result.setText(classes[maxpos]);
            time.setText(left[maxpos]);
            result.setTextColor(Color.parseColor(colours[maxpos]));  // Corrected line
            capture = findViewById(R.id.imageView4);
            cap = findViewById(R.id.capture);
            cap.setText("");
            capture.setAlpha(0);
            info.setText(inf[maxpos]);

            String st = "",sp = "";
            for (int i=0;i<classes.length;i++) {
                st += String.format("%s:\n", classes[i]);
                sp += String.format("%.1f%%\n", confidences[i] * 100);
            }

            String man = "Unripe: green, firm and may be difficult to peel\n" +
                    "\nRipe: yellow, yield slightly to gentle pressure and are easier to peel.\n" +
                    "\nOverripe: brown spots, mushy, and the peel may be brown and overly soft.\n";

            check.setText(man);
            con_text.setText(st);
            con_per.setText(sp);
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(),image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}