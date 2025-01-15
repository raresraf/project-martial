#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

char convert_to_lower(char x) {
  x = tolower(x);
  return x;
}

int main() {
  string a, b;
  cin >> a;
  cin >> b;
  char x[a.length()], y[b.length()];
  int same = 0;
  for (int i = 0; i < a.length(); i++) {
    x[i] = convert_to_lower(a[i]);
    y[i] = convert_to_lower(b[i]);
    if (x[i] != y[i]) {
      same = 1;
    }
  }
  if (same == 0) {
    cout << 0;
    return 0;
  }

  if (lexicographical_compare(x, x + 20, y, y + 20)) {
    cout << -1;
  }

  else if (lexicographical_compare(y, y + 20, x, x + 20)) {
    cout << 1;
  } else {
    cout << 0;
  }
}