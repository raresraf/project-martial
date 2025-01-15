#include <iostream>
#include <string>
using namespace std;

int main() {
  string arr;
  getline(cin, arr);
  bool ans = false;
  int len = arr.size(), co = 0, i;
  for (i = 0; i < len - 1; i++) {
    if (arr[i] == arr[i + 1]) {
      co++;
    } else {
      co = 0;
    }
    if (co >= 6) {
      ans = true;
    }
  }
  if (ans) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
