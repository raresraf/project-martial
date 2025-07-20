#include <iostream>
using namespace std;

int main() {
  int n, k, cont = 0, vec[100];
  cin >> n >> k;

  for (int i = 0; i < n; i++) {
    cin >> vec[i];
  }
  for (int j = 0; j < n; j++) {
    if (vec[j] >= vec[k - 1] & vec[j] > 0)
      cont++;
  }
  cout << cont;
}
