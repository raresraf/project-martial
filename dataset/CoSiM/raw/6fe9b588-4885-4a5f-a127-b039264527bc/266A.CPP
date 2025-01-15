#include <iostream>
// #include <algorithm>
// #include <vector>
// #include <set>
// #include <map>

using namespace std;

int main() {
  int n, count = 0;
  cin >> n;
  char prev = 0, curr;
  for (int i = 0; i < n; i++) {
    scanf(" %c", &curr);
    if (curr == prev)
      count++;

    prev = curr;
  }

  cout << count;
  return 0;
}
