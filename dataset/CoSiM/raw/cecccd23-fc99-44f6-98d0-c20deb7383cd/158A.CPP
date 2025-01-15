#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int n, k, c = 0, aux1, aux2;
  cin >> n >> k;
  for (; c < k; ++c) {
    cin >> aux1;
    if (aux1 == 0) {
      cout << c << "\n";
      return EXIT_SUCCESS;
    }
  }
  aux2 = aux1;
  while (c < n) {
    cin >> aux1;
    if (aux2 != aux1) {
      cout << c << "\n";
      return EXIT_SUCCESS;
    }
    c++;
  }
  cout << n << "\n";
}

/*
A. Next Round
time limit p/test: 3 seconds
memory limit p/test: 256 megabytes

"Contestant who earns a score equal to or greater than the k-th place finisher's
score will advance to the next round,
as long as the contestant earns a positive score..."

    Input::
The first line of the input contains two integers:
    n and k (1 <= k <= n <= 50) separated by a single space

The second line contains n space-separated integers,
where a_i is the score earned by the participant who got the i-th place.
Note: So it's sorted

Output the number of participants who advance to the next round
*/