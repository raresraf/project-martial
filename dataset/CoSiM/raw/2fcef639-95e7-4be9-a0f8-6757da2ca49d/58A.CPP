#include <iostream>
#include <string>
using namespace std;

int main() {
  int i, j = 0;
  string s = "hello";
  string x;
  cin >> x;
  for (i = 0; i < x.length(); i++) {
    if (x[i] == s[j]) {
      j++;
      cout << j << endl;
    }
  }
  if (j == 5) {
    cout << "YES";
  }

  else {
    cout << "NO";
  }

  return 0;
}