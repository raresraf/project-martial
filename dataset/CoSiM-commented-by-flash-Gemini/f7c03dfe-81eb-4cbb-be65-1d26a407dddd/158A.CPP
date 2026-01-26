#include <iostream>

using namespace std;

int main() {
  int contestant;
  int place;
  int counter = 0;
  int score;
  int lastscore;
  cin >> contestant;
  cin >> place;
  for (int i = 0; i < contestant; i++) {
    cin >> score;
    if (i < place && score > 0) {
      counter++;
      lastscore = score;
    } else if (score == lastscore) {
      counter++;
    } else {
      break;
    }
  }
  cout << counter;
  return 0;
}