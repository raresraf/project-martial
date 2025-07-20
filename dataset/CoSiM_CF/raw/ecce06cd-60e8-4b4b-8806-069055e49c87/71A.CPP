#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main() {

  // Input de # palabras a ingresar
  int n;
  cin >> n;

  // Declaro nuevo arreglo de n strings.
  string palabras[n];

  // Asigno n strings al arreglo palabras.
  for (int i = 0; i < n; i++) {
    string word;
    cin >> word;
    palabras[i] = word;
  }

  // Itero sobre cada elemento del arreglo palabras y trabajo con ellos
  for (int j = 0; j < n; j++) {
    if (palabras[j].length() > 10) {

      // Cuento # letras entre la primera y la  ltima.
      int count = 0;
      for (int k = 1; k < palabras[j].length() - 1; k++) {
        count++;
      }

      // Convierto esta cantidad a un string
      string result;
      stringstream sstm;
      sstm << count;
      result = sstm.str();

      // Modifico la palabra en la posicion j-esima por 'AcountZ'
      palabras[j] = palabras[j].replace(palabras[j].begin() + 1,
                                        palabras[j].end() - 1, result);
      cout << palabras[j] << endl;
    } else {
      cout << palabras[j] << endl;
    }
  }

  return 0;
}