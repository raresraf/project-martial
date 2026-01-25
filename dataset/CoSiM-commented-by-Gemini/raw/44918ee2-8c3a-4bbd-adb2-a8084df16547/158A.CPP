#include <iostream>
#include <new>
using namespace std;

int main() {
  int n, k, count = 0, thresh;
  int *score;
  cin >> n >> k;
  score = new int[n];
  for (int i = 0; i < n; i++) {
    cin >> score[i];
  }
  thresh = score[k - 1];
  for (int j = 0; j < n; j++) {
    if (score[j] >= thresh && score[j] > 0)
      count++;
  }
  cout << count;
  return 0;
}
