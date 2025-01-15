#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string x;
  int count = 0;
  cin >> x;

  for (int i = 0; i < x.length(); i++) {
    if (i == 0) {
      count++;
      continue;
    }
    if (x[i] == x[i - 1]) {
      count++;
      if (count >= 7)
        break;
    } else {
      count = 0;
      count++;
    }
  }
  if (count == 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
