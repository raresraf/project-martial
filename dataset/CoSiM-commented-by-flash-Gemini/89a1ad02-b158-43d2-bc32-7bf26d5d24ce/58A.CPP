#include <iostream>
using namespace std;
string a, b = "hello";
int j = 0, pass = 0;
int main() {
  cin >> a;
  for (int i = 0; i < a.size(); i++) {
    if (a[i] == b[j]) {
      j++;
      pass++;
    }
  }
  if (pass == 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}
