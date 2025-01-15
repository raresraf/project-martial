#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

int main() {
  string n;
  cin >> n;
  int count = 0;
  for (int i = 0; i < n.length(); ++i) {
    if (n[i] == '7' || n[i] == '4') {
      count += 1;
    }
  }
  if ((count == 7) || (count == 4)) {
    cout << "YES";
  } else {
    cout << "NO";
  }
}