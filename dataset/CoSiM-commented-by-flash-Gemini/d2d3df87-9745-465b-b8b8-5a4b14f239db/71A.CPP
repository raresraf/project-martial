#include <iostream>
#include <string>

using namespace std;

int main() {
  string palabra;
  int t, tam;
  cin >> t;
  while (t--) {
    cin >> palabra;
    tam = palabra.size();
    if (palabra.size() > 10) {
      cout << palabra.substr(0, 1) << tam - 2
           << palabra.substr(palabra.size() - 1, 1) << endl;
    } else {
      cout << palabra << endl;
    }
  }
  return 0;
}
