#include <bits/stdc++.h>

using namespace std;

int main() {
  int n;
  cin >> n;

  int a, b, capacity = 0, new_capacity = 0;

  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    new_capacity = (new_capacity - a) + b;
    if (new_capacity > capacity)
      capacity = new_capacity;
  }

  cout << capacity << endl;
}
