#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, max_capacity = 0, temp = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {
    int a, b;

    cin >> a >> b;

    temp = temp - a + b;

    if (temp > max_capacity)
      max_capacity = temp;
  }

  cout << max_capacity << endl;

  return 0;
}

/**
4
0 3
2 5
4 2
4 0

output:
6

special:
10
0 5
1 7
10 8
5 3
0 5
3 3
8 8
0 6
10 1
9 0

output:
18

**/
