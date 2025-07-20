#include <iostream>
#include <string>
using namespace std;
int main() {
  string cad1, cad2;
  cin >> cad1 >> cad2;
  bool aux = true;
  for (int i = 0; i < cad1.length(); i++) {
    if (tolower(cad1[i]) > tolower(cad2[i])) {
      cout << 1 << endl;
      aux = false;
      break;
    } else if (tolower(cad1[i]) < tolower(cad2[i])) {
      cout << -1 << endl;
      aux = false;
      break;
    }
  }
  if (aux)
    cout << 0 << endl;
  return 0;
}
