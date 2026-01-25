#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {

  string cad;
  map<char, int> caracteres;
  cin >> cad;
  for (int i = 0; i < cad.size(); i++) {
    caracteres[cad[i]]++;
  }

  if (caracteres.size() & 1) {
    cout << "IGNORE HIM!" << endl;
  } else {
    cout << "CHAT WITH HER!" << endl;
  }

  return 0;
}