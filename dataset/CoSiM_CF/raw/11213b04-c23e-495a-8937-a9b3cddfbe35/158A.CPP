#include <iostream>
#include <vector>
using namespace std;

int main() {
  int count, place, element, flag = -1;
  vector<int> scores;

  cin >> count >> place;

  for (int k = 0; k < count; k++) {
    cin >> element;
    scores.push_back(element);

    if (flag == -1 && element == 0)
      flag = k;
  }

  if (flag != -1 && scores[flag] == 0 && flag < place) {
    cout << flag;
    return 0;
  }
  while (true) {
    if ((count != place) && (scores[place - 1] == scores[place])) {
      place++;
    } else {
      break;
    }
  }

  cout << place;
  return 0;
}