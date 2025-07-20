#include <bits/stdc++.h>
using namespace std;

pair<int, int> num_lucky(long long x) {
  int amount = 0;
  int len = x == 0;
  while (x > 0) {
    int d = x % 10;
    if (d == 4 || d == 7)
      amount++;
    len++;
    x /= 10;
  }
  return make_pair(amount, len);
}

int main(int argc, char *argv[]) {
  long long x;
  cin >> x;

  auto l = num_lucky(num_lucky(x).first);

  cout << (l.first == l.second ? "YES" : "NO") << endl;
}
