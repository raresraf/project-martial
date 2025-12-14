#include <bits/stdc++.h>

using namespace std;

int main() {

  int a, b;
  int n;

  int max_passenger = -1;
  int curr_passengers = 0;
  ;

  scanf("%d", &n);

  while (n--) {

    scanf("%d%d", &a, &b);

    curr_passengers -= a;
    curr_passengers += b;

    if (max_passenger < curr_passengers)
      max_passenger = curr_passengers;
  }

  cout << max_passenger << endl;
}