#include <iostream>
#include <string>
using namespace std;

bool periksa(string kal) {

  string kata = "hello";
  int curr = -1;
  for (int i = 0; i < kal.length(); i++) {
    if (kal[i] == kata[curr + 1]) {
      curr++;
      if (curr == kata.length() - 1) {
        return true;
      }
    }
  }
  return false;
}

int main() {

  string klmt;

  cin >> klmt;
  if (periksa(klmt)) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
