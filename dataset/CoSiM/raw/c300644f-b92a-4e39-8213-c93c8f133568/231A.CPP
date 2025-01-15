#include <bits/stdc++.h>

using namespace std;

int main() {

  int n;
  int a, b, c;
  int count = 0;

  scanf("%d", &n);

  while (n--) {

    scanf("%d%d%d", &a, &b, &c);

    if (a & b || b & c || c & a)
      count++;
  }

  cout << count << endl;
}