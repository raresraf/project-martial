#include <iostream>
#include <vector>
using namespace std;
int main() {
  typedef long long ll;
  ll n, count = 0, t = 0, p1, p2, p3;
  cin >> n;
  while (n--) {
    t = 0;
    cin >> p1 >> p2 >> p3;
    t = p1 + p2 + p3;
    if (t >= 2)
      count++;
  }
  cout << count << "\n";
}