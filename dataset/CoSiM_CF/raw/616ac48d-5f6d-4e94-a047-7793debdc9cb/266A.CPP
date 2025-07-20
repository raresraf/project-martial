#include <iostream>
#include <string>
using namespace std;

int main() {
  int n, i, count = 0;
  string s;
  cin >> n;
  for (i = 0; i < n; i++) {
    cin >> s;
    if (s[i] == s[i + 1]) {
      count++;
    }
  }
  if (count == 0) {
    cout << "0" << endl;
  } else {
    cout << count << endl;
  }
  return 0;
}
