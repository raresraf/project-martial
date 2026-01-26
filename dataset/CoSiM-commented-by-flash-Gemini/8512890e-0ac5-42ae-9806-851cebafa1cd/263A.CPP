#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
  int a[5][5];
  int posx, posy;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; ++j) {
      cin >> a[i][j];
      if (a[i][j] > 0) {
        posx = i;
        posy = j;
      }
    }
  }
  int answer_posx = pow(posx - 2, 2);
  int answer_posy = pow(posy - 2, 2);
  answer_posx = sqrt(answer_posx);
  answer_posy = sqrt(answer_posy);
  int answer = answer_posx + answer_posy;
  cout << answer;
}