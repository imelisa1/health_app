import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() => runApp(MaterialApp(home: HealthAnalyzer()));

class HealthAnalyzer extends StatefulWidget {
  @override
  _HealthAnalyzerState createState() => _HealthAnalyzerState();
}

class _HealthAnalyzerState extends State<HealthAnalyzer> {
  Interpreter? interpreter;
  File? imageFile;
  String resultText = "";
  final picker = ImagePicker();

  Future<void> pickAndPredict(String assetPath, String modelType) async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile == null) return;

    imageFile = File(pickedFile.path);
    final imageBytes = await imageFile!.readAsBytes();
    final oriImage = img.decodeImage(imageBytes);
    final resized = img.copyResize(oriImage!, width: 224, height: 224);

    final input = Float32List(1 * 224 * 224 * 3);
    int idx = 0;
    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        input[idx++] = img.getRed(pixel) / 255.0;
        input[idx++] = img.getGreen(pixel) / 255.0;
        input[idx++] = img.getBlue(pixel) / 255.0;
      }
    }

    final inputTensor = input.reshape([1, 224, 224, 3]);

    interpreter = await Interpreter.fromAsset(assetPath);

    if (modelType == 'sigmoid') {
      var output = List.filled(1, 0.0).reshape([1, 1]);
      interpreter!.run(inputTensor, output);
      double score = output[0][0];
      String label;
      if (assetPath.contains("lung")) {
        label = score > 0.5 ? "Pneumonia" : "Normal";
      } else if (assetPath.contains("skin")) {
        label = score > 0.5 ? "Malignant" : "Benign";
      } else if (assetPath.contains("fracture")) {
        label = score > 0.5 ? "Fractured" : "Not Fractured";
      } else {
        label = score > 0.5 ? "Not Fructured" : "Fructured";
      }
      setState(() {
        resultText = "$label (${(score * 100).toStringAsFixed(2)}%)";
      });
    } else if (modelType == 'softmax') {
      var output = List.filled(3, 0.0).reshape([1, 3]);
      interpreter!.run(inputTensor, output);
      final scores = output[0];
      final labels = ['Glioma', 'Meningioma', 'Pituitary'];
      double maxScore = scores.reduce((double a, double b) => a > b ? a : b);
      int maxIdx = scores.indexOf(maxScore);

      setState(() {
        resultText =
            '${labels[maxIdx]} (${(scores[maxIdx] * 100).toStringAsFixed(2)}%)';
      });
    }
  }

  Widget buildButton(String label, String path, String modelType) {
    return ElevatedButton(
      onPressed: () => pickAndPredict(path, modelType),
      child: Text(label),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Health Analyzer"), centerTitle: true),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            buildButton("Cilt Analizi", "skin.tflite", "sigmoid"),
            buildButton("Beyin Tümörü", "brain.tflite", "softmax"),
            buildButton("Zatürre Tespiti", "lung.tflite", "sigmoid"),
            buildButton("Kırık Tespiti", "bone.tflite", "sigmoid"),
            SizedBox(height: 20),
            imageFile != null
                ? Image.file(imageFile!, height: 200)
                : Container(height: 200, color: Colors.grey[300]),
            SizedBox(height: 10),
            Text(resultText, style: TextStyle(fontSize: 18)),
          ],
        ),
      ),
    );
  }
}
