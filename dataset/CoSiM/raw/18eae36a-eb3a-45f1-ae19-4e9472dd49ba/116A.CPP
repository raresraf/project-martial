#include <iostream>
using namespace std;
int main() {
  int iter(0), n(0);
  cin >> n;
  int enter[n], exit[n], final[n], min_capacity(0);
  for (iter = 0; iter < n; iter++) {
    cin >> enter[iter] >> exit[iter];
  }
  for (iter = 0; iter < n; iter++) {
    if (iter == 0) {
      final[iter] = enter[iter] + exit[iter];
    } else {
      final[iter] = final[iter - 1] - enter[iter] + exit[iter];
    }
  }
  for (iter = 0; iter < n; iter++) {
    if (min_capacity < final[iter]) {
      min_capacity = final[iter];
    } else {
      continue;
    }
  }
  cout << min_capacity;
  return 0;
}