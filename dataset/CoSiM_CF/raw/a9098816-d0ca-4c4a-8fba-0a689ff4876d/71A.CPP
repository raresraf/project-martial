#include <iostream>
//#include <cstring>
//#include <string>
//#include <sstream>
//#include <stdlib.h>
using namespace std;

int main() {
  int n;
  string palabra;
  // char aux[2];
  // string vector[100];
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> palabra;
    if (palabra.length() > 10) {
      // itoa(palabra.length()-2,aux,10);
      // vector[i] = palabra[0]+string(aux)+palabra[palabra.length()-1];
      cout << palabra[0] << palabra.length() - 2
           << palabra[palabra.length() - 1] << endl;
    } else {
      // vector[i]= palabra;
      cout << palabra << endl;
    }
  }
}
