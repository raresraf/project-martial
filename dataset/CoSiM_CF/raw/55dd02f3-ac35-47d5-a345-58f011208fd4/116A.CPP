#include <bits/stdc++.h>
using namespace std;

int main() {
  int N, sum = 0, capa = 0;
  cin >> N;
  int Enter[N + 1], Exit[N + 1];
  for (int i = 0; i < N; i++)
    cin >> Exit[i] >> Enter[i];
  sum = fabs(Enter[0] - Exit[0]);
  if (capa < sum)
    capa = sum;
  for (int i = 1; i < N; i++) {
    sum -= Exit[i];
    sum += Enter[i];
    if (capa < sum)
      capa = sum;
  }
  cout << capa << endl;

  return 0;
}
