// CodeForces
// Problem 158A [Next Round]

#include <iostream>
using namespace std;

int main() {
  unsigned int total_participants = 0, qualifying_position = 0,
               total_qualifiers = 0;
  cin >> total_participants >> qualifying_position;

  int scores_array[51];
  for (unsigned int input_index = 0; input_index < total_participants;
       input_index++) {
    cin >> scores_array[input_index];
  }

  for (unsigned int checker_index = 0; checker_index < total_participants;
       checker_index++) {
    if ((scores_array[checker_index] >=
         scores_array[qualifying_position - 1]) &&
        (scores_array[checker_index] > 0)) {
      total_qualifiers++;
    }
  }
  cout << total_qualifiers;
  return 0;
}