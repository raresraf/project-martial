#include <bits/stdc++.h>
using namespace std;

int main() {

  long long int cost, mon, need, tot = 0;
  cin >> cost >> mon >> need;

  for (int i = 1; i <= need; i++) {
    tot += i * cost;
  }
  if (tot < mon)
    cout << "0" << endl;
  else
    cout << tot - mon << endl;

  return 0;
}
