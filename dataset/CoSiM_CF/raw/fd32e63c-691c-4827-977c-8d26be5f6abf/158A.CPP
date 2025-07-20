#include <iostream>

using namespace std;
int main() {
  int n, k, passingScore, totalPassing;
  cin >> n >> k;

  int scores[51];
  for (int i = 0; i < n; i++) {
    cin >> scores[i];
  }

  passingScore = scores[k - 1];

  totalPassing = 0;
  for (int i = 0; i < n; i++) {
    if (scores[i] > 0 && scores[i] >= passingScore)
      totalPassing++;
  }

  cout << totalPassing;
}
