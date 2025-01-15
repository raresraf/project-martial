#include <algorithm>
#include <iostream>

using namespace std;

int main() {
  int n, count = 0, line_count, i, temp;
  cin >> n;
  while (n--) {
    line_count = 0;
    for (i = 0; i < 3; i++) {
      cin >> temp;
      line_count += temp;
    }
    if (line_count >= 2)
      count++;
  }
  cout << count << endl;
  return 0;
}