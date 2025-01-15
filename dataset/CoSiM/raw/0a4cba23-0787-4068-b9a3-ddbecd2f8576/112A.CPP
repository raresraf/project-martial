#include <iostream>
#include <string>
using namespace std;

int main() {
  bool dis = true;
  string kata1, kata2;
  cin >> kata1;
  cin >> kata2;

  for (int i = 0; i < kata1.length(); i++) {
    if (kata1[i] >= 'A' && kata1[i] <= 'Z') {
      kata1[i] = kata1[i] - 'A' + 'a';
    }
    if (kata2[i] >= 'A' && kata2[i] <= 'Z') {
      kata2[i] = kata2[i] - 'A' + 'a';
    }

    if (kata1[i] < kata2[i]) {
      dis = false;
      cout << -1;
      break;
    } else if (kata1[i] > kata2[i]) {
      dis = false;
      cout << 1;
      break;
    }
  }
  if (dis) {
    cout << 0;
  }
  return 0;
}
